import fiona
from collections import defaultdict
from shapely.geometry import shape, Point, LineString, MultiLineString
from rtree import index


class Edge(object):

    @classmethod
    def from_feature(cls, network, feature):
        return cls(network, int(feature['id']), shape(feature['geometry']))

    def __init__(self, network, id, geometry, start_node=None, end_node=None):
        self.network = network
        self.id = id
        self.geometry = geometry
        self.start_node = start_node
        self.end_node = end_node

    def __repr__(self):
        return '<Edge: %i>' % (self.id,)

    @property
    def start_coords(self):
        return self.geometry.coords[0]

    @property
    def end_coords(self):
        return self.geometry.coords[-1]

    def other_node(self, node):
        if self.start_node == node:
            return self.end_node
        return self.start_node


class Node(object):

    def __init__(self, network, id, geometry=None, edges=None):
        self.network = network
        self.id = id
        self.geometry = geometry
        self.edges = edges or []

    def __repr__(self):
        return '<Node: %i>' % (self.id,)

    def distance(self, other):
        return self.geometry.distance(other.geometry)

    def to_feature(self):
        return {
            'geometry': self.geometry,
            'properties': {
                'id': self.id
            }
        }


class Network(object):

    def __init__(self, path, layer_name):
        self.next_node_id = 1
        self.edges = self._get_edges(path, layer_name)
        self.nodes = self._get_nodes()
        self.nodes_idx = self._make_spatial_index(self.nodes)

    def __repr__(self):
        return '<Network>'

    def _get_edges(self, path, layer_name):
        with fiona.open(path, layer=layer_name) as source:
            self.meta = source.meta
            return [Edge.from_feature(self, f) for f in source]

    def _get_nodes(self):
        def make_node():
            node = Node(self, self.next_node_id)
            self.next_node_id += 1
            return node

        coords_node = defaultdict(make_node)
        for edge in self.edges:
            for position in ['start', 'end']:
                coords = getattr(edge, position + '_coords')
                node = coords_node[coords]
                setattr(edge, position + '_node', node)
                if not node.geometry:
                    node.geometry = Point(coords)
                node.edges.append(edge)

        return list(coords_node.values())

    def _make_spatial_index(self, items):
        idx = index.Index()
        for (item_idx, item) in enumerate(items):
            idx.insert(item_idx, item.geometry.bounds)
        return idx

    def get_nearest_node(self, node, threshold=0):
        nearest = [self.nodes[i] for i in
                   self.nodes_idx.nearest(node.geometry.bounds, 1)]

        if not nearest or \
                (threshold > 0 and nearest[0].distance(node) > threshold):
            return None

        return nearest[0]


class Matches(object):

    def __init__(self, a_network, b_network, attr):
        self.a_network = a_network
        self.b_network = b_network
        self.a_map = self._get_id_map(getattr(a_network, attr))
        # self.b_map = self._get_id_map(getattr(b_network, attr))
        self.ab = {}
        self.ba = {}

    def _get_id_map(self, objs):
        return dict([(obj.id, obj) for obj in objs])

    def add(self, a, b):
        if a.id not in self.ab:
            self.ab[a.id] = []

        if b.id not in self.ba:
            self.ba[b.id] = []

        self.ab[a.id].append(b)
        self.ba[b.id].append(a)

    def get(self, obj):
        if obj.network == self.a_network:
            return self.ab.get(obj.id, [])
        return self.ba.get(obj.id, [])

    def items(self):
        for (a_id, b_list) in self.ab.items():
            a = self.a_map[a_id]
            for b in b_list:
                yield (a, b)

    def has_match(self, obj):
        return len(self.get(obj)) > 0

    def matches(self, obj, other):
        return other in self.get(obj)


class Sequence(object):

    def __init__(self, start_node, end_node=None, edges=None):
        self.start_node = start_node
        self.end_node = end_node
        self.edges = edges or []

    def __repr__(self):
        return '<Sequence: %i to %i>' % (
            getattr(self.start_node, 'id', -1),
            getattr(self.end_node, 'id', -1))

    @property
    def count(self):
        return len(self.edges)

    @property
    def length(self):
        return sum([e.geometry.length for e in self.edges])

    @property
    def geometry(self):
        return MultiLineString([e.geometry for e in self.edges])

    def add(self, edge):
        end_node = edge.other_node(self.end_node or self.start_node)
        return Sequence(self.start_node, end_node, self.edges + [edge])

    def includes(self, edge):
        return edge in self.edges

    def hdistance(self, other):
        return self.geometry.hausdorff_distance(other.geometry)


class Matcher(object):

    INITIAL_NODE_THRESHOLD = 50
    EDGE_SEARCH_LIMIT = 3
    EDGE_MAX_HAUSDORFF_DISTANCE = 100

    def __init__(self, a_network, b_network):
        for net in [a_network, b_network]:
            assert isinstance(net, Network), \
                'arguments must be an instances of Network'

        self.a_network = a_network
        self.b_network = b_network
        self.node_matches = Matches(a_network, b_network, 'nodes')
        self.edge_matches = Matches(a_network, b_network, 'edges')

    def _match_nodes_by_proximity(self, max_dist=0):
        for a_node in self.a_network.nodes:
            b_node = self.b_network.get_nearest_node(a_node, max_dist)
            if b_node is not None and self.a_network.get_nearest_node(
                    b_node, max_dist) == a_node:
                self.node_matches.add(a_node, b_node)

    def _iter_possible_edge_sequences(self, node, limit, seq=None):
        if seq is None:
            seq = Sequence(node)

        for edge in sorted(node.edges, key=lambda e: e.geometry.length):
            if seq.includes(edge) or self.edge_matches.has_match(edge):
                continue

            new_seq = seq.add(edge)
            if self.node_matches.has_match(new_seq.end_node):
                yield new_seq
            elif limit > 0:
                for child_seq in self._iter_possible_edge_sequences(
                        new_seq.end_node, limit - 1, new_seq):
                    yield child_seq

    def _match_edges_from_matched_nodes(self, search_limit, max_distance):
        for (a_node, b_node) in self.node_matches.items():
            for a_seq in self._iter_possible_edge_sequences(
                    a_node, search_limit):
                for b_seq in self._iter_possible_edge_sequences(
                        b_node, search_limit):
                    if self.node_matches.matches(
                            a_seq.end_node, b_seq.end_node):
                        if a_seq.hdistance(b_seq) < max_distance:
                            self._match_sequence_edges(a_seq, b_seq)

    def _match_sequence_edges(self, a_seq, b_seq):
        # TODO: This is not really correct. We need to compare the edge
        # geometries in the sequences and only match ones where they are
        # parallel to each other.
        for a_edge in a_seq.edges:
            for b_edge in b_seq.edges:
                self.edge_matches.add(a_edge, b_edge)

    def export_node_match_lines(self, out_path, out_layer):
        meta = self.a_network.meta.copy()
        meta['schema'] = {
            'geometry': 'LineString',
            'properties': {
                'a_node': 'int',
                'b_node': 'int'
            }
        }
        with fiona.open(out_path, 'w', layer=out_layer, **meta) as dest:
            for (a, b) in self.node_matches.items():
                geom = LineString([a.geometry, b.geometry])
                dest.write({
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': list(geom.coords)
                    },
                    'properties': {
                        'a_node': a.id,
                        'b_node': b.id
                    }
                })

    def export_edge_match_results(self, out_path, out_layer, network):
        meta = network.meta.copy()
        meta['schema'] = {
            'geometry': 'LineString',
            'properties': {
                'id': 'int',
                'match_count': 'int',
                'match_ids': 'str',
            }
        }
        with fiona.open(out_path, 'w', layer=out_layer, **meta) as dest:
            for edge in network.edges:
                matches = self.edge_matches.get(edge)
                dest.write({
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': list(edge.geometry.coords)
                    },
                    'properties': {
                        'id': edge.id,
                        'match_count': len(matches),
                        'match_ids': ', '.join([str(m.id) for m in matches])
                    }
                })

    def match(self):
        self._match_nodes_by_proximity(self.INITIAL_NODE_THRESHOLD)
        self._match_edges_from_matched_nodes(
            self.EDGE_SEARCH_LIMIT, self.EDGE_MAX_HAUSDORFF_DISTANCE)


if __name__ == '__main__':
    print('Creating networks...')
    a_net = Network('network.gpkg', 'a_edge')
    b_net = Network('network.gpkg', 'b_edge')

    print('Matching networks...')
    matcher = Matcher(a_net, b_net)
    matcher.match()

    print('Exporting results...')
    matcher.export_node_match_lines('matches.gpkg', 'node_matches')
    matcher.export_edge_match_results('matches.gpkg', 'a_edge', a_net)
    matcher.export_edge_match_results('matches.gpkg', 'b_edge', b_net)
