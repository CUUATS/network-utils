import math
import time
from collections import defaultdict
from qgis.core import QgsApplication, QgsVectorLayer, QgsGeometry, \
    QgsSpatialIndex, QgsField, edit, QgsFeature
from PyQt5.QtCore import QVariant


class timeable(object):

    def __init__(self, action):
        self._action = action

    def __call__(self, method):

        def timed(*args, **kwargs):
            show_time = kwargs.get('show_time', False)
            if 'show_time' in kwargs:
                del kwargs['show_time']

            if not show_time:
                return method(*args, **kwargs)

            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()

            print('%s in %0.3f sec' % (self._action, end_time - start_time))
            return result

        return timed


class Network(object):

    FIELDS = [
        ('matches', QVariant.String),
        ('matches_count', QVariant.Int)]

    def __init__(self, path, index_nodes=True):
        self._path = path
        self._layer = QgsVectorLayer(path)

        self._next_edge_id = -1
        self._edge_map = dict([(f.id(), f) for f in self._layer.getFeatures()])
        self._edge_index = QgsSpatialIndex()
        self._edge_nodes = {}

        self._node_map = {}
        self._node_index = QgsSpatialIndex()
        self._node_edges = defaultdict(set)

        self._build_indexes(index_nodes)

    def __repr__(self):
        return '<Network %s>' % (self._path,)

    def _build_indexes(self, index_nodes):
        next_node_id = 1
        coords_node = {}

        for (fid, feature) in self._edge_map.items():
            fid = feature.id()
            ls = feature.geometry().constGet()
            endpoints = []
            for point in [ls.startPoint(), ls.endPoint()]:
                coords = (point.x(), point.y())
                node_id = coords_node.get(coords, None)

                if node_id is None:
                    node_id = next_node_id
                    next_node_id += 1
                    coords_node[coords] = node_id
                    node_feature = self._make_feature(
                        node_id, QgsGeometry(point))
                    self._node_map[node_id] = node_feature
                    if index_nodes:
                        self._node_index.insertFeature(node_feature)

                endpoints.append(node_id)
                self._node_edges[node_id].add(fid)

            self._edge_index.insertFeature(feature)
            self._edge_nodes[fid] = endpoints

        self._next_edge_id = fid + 1

    def _has_field(self, field_name):
        return field_name in [f.name() for f in self._layer.fields()]

    def _add_fields(self, fields):
        self._layer.dataProvider().addAttributes(
            [QgsField(fn, ft) for (fn, ft) in fields])

    def _make_feature(self, fid, geometry):
        feature = QgsFeature(fid)
        feature.setGeometry(geometry)
        return feature

    def _vertex_id(self, eid, nid):
        if self.get_edge_nids(eid)[0] == nid:
            return 0
        return self.get_edge(eid).geometry().constGet().nCoordinates() - 1

    def _node_angle(self, eid, nid):
        vid = self._vertex_id(eid, nid)
        angle = self.get_edge(eid).geometry().angleAtVertex(vid) \
            * 180 / math.pi
        if vid > 0:
            angle += 180 if angle < 180 else -180
        return angle

    def eids(self):
        return self._edge_map.keys()

    def nids(self):
        return self._node_map.keys()

    def get_edge(self, eid):
        return self._edge_map[eid]

    def get_node(self, nid):
        return self._node_map[nid]

    def find_nids(self, bbox):
        return self._node_index.intersects(bbox)

    def find_eids(self, bbox):
        return self._edge_index.intersects(bbox)

    def get_edge_nids(self, eid):
        return self._edge_nodes[eid]

    def get_node_eids(self, nid):
        return self._node_edges[nid]

    def get_other_nid(self, eid, nid):
        nids = self._edge_nodes[eid]
        if nids[0] == nid:
            return nids[1]
        return nids[0]

    def get_node_angles(self, nid):
        eids = self.get_node_eids(nid)
        return dict([(eid, self._node_angle(eid, nid)) for eid in eids])

    def is_loop(self, eid):
        return len(set(self.get_edge_nids(eid))) == 1

    def write_matches(self, match_dict):
        with edit(self._layer):
            new_fields = [(fn, ft) for (fn, ft) in self.FIELDS
                          if not self._has_field(fn)]
            if new_fields:
                self._add_fields(new_fields)
                self._layer.updateFields()

            provider = self._layer.dataProvider()
            matches_idx = provider.fieldNameIndex('matches')
            count_idx = provider.fieldNameIndex('matches_count')

            changes = defaultdict(dict)
            for fid in self._edge_map.keys():
                matches = match_dict.get(fid, [])
                changes[fid][matches_idx] = \
                    ', '.join([str(m) for m in matches])
                changes[fid][count_idx] = len(matches)

            provider.changeAttributeValues(dict(changes))


class Matcher(object):

    @classmethod
    @timeable('Created networks')
    def from_paths(cls, a_path, b_path, **kwargs):
        return cls(Network(a_path, index_nodes=False),
                   Network(b_path), **kwargs)

    def __init__(self, a_network, b_network, **kwargs):
        for net in [a_network, b_network]:
            assert isinstance(net, Network), \
                'arguments must be an instances of Network'

        self._a_network = a_network
        self._b_network = b_network

        self._ab_node_node = {}
        self._b_node_node = set()

        self._ab_node_edge = defaultdict(set)
        self._ba_node_edge = defaultdict(set)
        self._ab_node_edge_dist = {}
        self._ba_node_edge_dist = {}

        self._ab_edge_edge = defaultdict(set)
        self._ba_edge_edge = defaultdict(set)

        self._distance = kwargs.get('distance', 100)
        self._segments = kwargs.get('segments', 20)

    def _networks(self):
        yield (self._a_network, self._b_network)
        yield (self._b_network, self._a_network)

    def _choose(self, network, a, b):
        return a if network == self._a_network else b

    def _match_nodes_to_nodes(self):
        ab = {}
        ba = {}
        a_dist = {}
        b_dist = {}
        default_dist = self._distance + 1

        for a_nid in self._a_network.nids():
            a_node = self._a_network.get_node(a_nid)
            bbox = a_node.geometry().boundingBox().buffered(self._distance)
            for b_nid in self._b_network.find_nids(bbox):
                b_node = self._b_network.get_node(b_nid)
                distance = a_node.geometry().distance(b_node.geometry())
                if distance <= self._distance:
                    if distance < a_dist.get(a_nid, default_dist):
                        ab[a_nid] = b_nid
                        a_dist[a_nid] = distance
                    if distance < b_dist.get(b_nid, default_dist):
                        ba[b_nid] = a_nid
                        b_dist[b_nid] = distance

        for (a_nid, b_nid) in ab.items():
            if ba.get(b_nid, None) == a_nid:
                self._ab_node_node[a_nid] = b_nid
                self._b_node_node.add(b_nid)

    def _match_nodes_to_edges(self):
        for (network, other_network) in self._networks():
            for nid in network.nids():
                node = network.get_node(nid)
                bbox = node.geometry().boundingBox().buffered(self._distance)
                for eid in other_network.find_eids(bbox):
                    edge = other_network.get_edge(eid)
                    distance = node.geometry().distance(edge.geometry())
                    if distance <= self._distance:
                        node_edge = self._choose(
                            network, self._ab_node_edge, self._ba_node_edge)
                        dist = self._choose(network, self._ab_node_edge_dist,
                                            self._ba_node_edge_dist)
                        node_edge[nid].add(eid)
                        dist[(nid, eid)] = distance

    def _match_edges_to_edges(self):
        for (a_nid, b_nid) in self._ab_node_node.items():
            for a_eid in self._a_network.get_node_eids(a_nid):
                if a_eid in self._ab_edge_edge:
                    continue
                a_end_nid = self._a_network.get_other_nid(a_eid, a_nid)
                for b_eid in self._b_network.get_node_eids(b_nid):
                    if b_eid in self._ba_edge_edge:
                        continue
                    b_end_nid = self._b_network.get_other_nid(b_eid, b_nid)
                    if self._find_edge_to_edge_match(
                            a_end_nid, b_end_nid, a_eid, b_eid):
                        break

    def _find_edge_to_edge_match(self, a_end_nid, b_end_nid, a_eid, b_eid):
        for matches in self._iter_edge_matches(
                a_end_nid, b_end_nid, [a_eid], [b_eid], [], self._segments):
            for (a_eid, b_eid) in matches:
                self._ab_edge_edge[a_eid].add(b_eid)
                self._ba_edge_edge[b_eid].add(a_eid)
            # Only find one matching sequence per edge pair.
            return True
        return False

    def _state_endpoint_distance(self, a_nid, b_nid, a_eids, b_eids):
        default_dist = self._distance + 1
        ab_dist = self._ab_node_edge_dist.get(
            (a_nid, b_eids[-1]), default_dist)
        ba_dist = self._ba_node_edge_dist.get(
            (b_nid, a_eids[-1]), default_dist)
        return min(ab_dist, ba_dist)

    def _get_next_states(self, a_nid, b_nid, a_eids, b_eids):
        states = []
        if b_eids[-1] in self._ab_node_edge.get(a_nid, set()) and \
                a_nid not in self._ab_node_node:
            for a_eid in self._a_network.get_node_eids(a_nid):
                if a_eid in a_eids or a_eid in self._ab_edge_edge:
                    continue
                a_other_nid = self._a_network.get_other_nid(a_eid, a_nid)
                a_new_eids = a_eids + [a_eid]
                states.append((a_other_nid, b_nid, a_new_eids, b_eids))
        if a_eids[-1] in self._ba_node_edge.get(b_nid, set()) and \
                b_nid not in self._b_node_node:
            for b_eid in self._b_network.get_node_eids(b_nid):
                if b_eid in b_eids or b_eid in self._ba_edge_edge:
                    continue
                b_other_nid = self._b_network.get_other_nid(b_eid, b_nid)
                b_new_eids = b_eids + [b_eid]
                states.append((a_nid, b_other_nid, a_eids, b_new_eids))

        state_dist = [
            (self._state_endpoint_distance(*state), state) for state in states]
        return [s for (d, s) in sorted(state_dist) if d <= self._distance]

    def _iter_edge_matches(self, a_nid, b_nid, a_eids, b_eids, matches,
                           retries):
        matches = matches + [(a_eids[-1], b_eids[-1])]
        if self._ab_node_node.get(a_nid, None) == b_nid:
            if self._sequence_hausdorff_distance(a_eids, b_eids) <= \
                    self._distance:
                yield matches
        elif retries > 0:
            for (a_nid, b_nid, a_eids, b_eids) in self._get_next_states(
                    a_nid, b_nid, a_eids, b_eids):
                for matches in self._iter_edge_matches(
                        a_nid, b_nid, a_eids, b_eids, matches, retries - 1):
                    yield matches

    def _sequence_hausdorff_distance(self, a_eids, b_eids):
        a_geom = self._combine_feature_geometries(
            [self._a_network.get_edge(eid) for eid in a_eids])
        b_geom = self._combine_feature_geometries(
            [self._b_network.get_edge(eid) for eid in b_eids])

        return a_geom.hausdorffDistance(b_geom)

    def _combine_feature_geometries(self, features):
        geom = features.pop().geometry()
        while features:
            geom = geom.combine(features.pop().geometry())
        return geom

    @timeable('Matched')
    def match(self):
        self._match_nodes_to_nodes()
        self._match_nodes_to_edges()
        self._match_edges_to_edges()

    @timeable('Wrote matches')
    def write_matches(self):
        self._a_network.write_matches(self._ab_edge_edge)
        self._b_network.write_matches(self._ba_edge_edge)


if __name__ == '__main__':
    qgs = QgsApplication([], False)
    qgs.initQgis()

    print('Creating networks...')
    matcher = Matcher.from_paths(
        'network.gpkg|layername=a_edge', 'network.gpkg|layername=b_edge',
        show_time=True)

    print('Matching...')
    matcher.match(show_time=True)

    print('Writing matches...')
    matcher.write_matches(show_time=True)

    print('A-B edge matches: %i ' % (len(matcher._ab_edge_edge),))
    print('B-A edge matches: %i ' % (len(matcher._ba_edge_edge),))

    qgs.exitQgis()
