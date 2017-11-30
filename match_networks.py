import time
from collections import defaultdict
from qgis.core import QgsApplication, QgsVectorLayer, QgsGeometry, \
    QgsSpatialIndex, QgsField, edit, QgsFeature
from PyQt5.QtCore import QVariant


class Network(object):

    def __init__(self, path):
        self._path = path
        self._layer = QgsVectorLayer(path)

        self._next_edge_id = -1
        self._edge_map = dict([(f.id(), f) for f in self._layer.getFeatures()])
        self._edge_index = QgsSpatialIndex()
        self._edge_nodes = {}

        self._node_map = {}
        self._node_index = QgsSpatialIndex()
        self._node_edges = defaultdict(set)

        self._build_indexes()

    def __repr__(self):
        return '<Network %s>' % (self._path,)

    def _build_indexes(self):
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

    def is_loop(self, eid):
        return len(set(self.get_edge_nids(eid))) == 1

    def remove_edge(self, eid):
        # print('Removing edge %i' % (eid,))
        edge = self.get_edge(eid)
        for nid in list(self.get_edge_nids(eid)):
            self._node_edges[nid] -= set([eid])
        self._edge_index.deleteFeature(edge)
        del self._edge_nodes[eid]
        del self._edge_map[eid]

    def remove_node(self, nid):
        # print('Removing node %i' % (nid,))
        node = self.get_node(nid)
        for eid in list(self.get_node_eids(nid)):
            self.remove_edge(eid)
        self._node_index.deleteFeature(node)
        del self._node_edges[nid]
        del self._node_map[nid]

    def merge_edges(self, m_eid, n_eid):
        m_nids = set(self.get_edge_nids(m_eid))
        n_nids = set(self.get_edge_nids(n_eid))
        shared_nids = m_nids & n_nids

        assert len(shared_nids) == 1, \
            'Edges %i and %i do not share a common node' % (m_eid, n_eid)

        old_nid = shared_nids.pop()
        endpoints = list(m_nids ^ n_nids)
        new_eid = self._next_edge_id
        self._next_edge_id += 1

        m_edge = self.get_edge(m_eid)
        n_edge = self.get_edge(n_eid)
        geometry = m_edge.geometry().combine(n_edge.geometry())
        geometry.mergeLines()
        feature = self._make_feature(self._next_edge_id, geometry)

        self.remove_node(old_nid)
        self._edge_map[new_eid] = feature
        self._edge_index.insertFeature(feature)
        self._edge_nodes[new_eid] = endpoints
        for nid in endpoints:
            self._node_edges[nid].add(new_eid)
        return new_eid

    def write_matches(self, match_dict):
        with edit(self._layer):
            if not self._has_field('matches'):
                self._add_fields([
                    ('matches', QVariant.String),
                    ('matches_count', QVariant.Int)])
                self._layer.updateFields()

            provider = self._layer.dataProvider()
            matches_idx = provider.fieldNameIndex('matches')
            count_idx = provider.fieldNameIndex('matches_count')

            changes = {}
            for fid in self._map.keys():
                matches = match_dict[fid]
                feature_changes = {}
                feature_changes[matches_idx] = \
                    ', '.join([str(m) for m in matches])
                feature_changes[count_idx] = len(matches)
                changes[fid] = feature_changes

            provider.changeAttributeValues(changes)


class Matcher(object):

    def __init__(self, a_network, b_network, **kwargs):
        for net in [a_network, b_network]:
            assert isinstance(net, Network), \
                'arguments must be an instances of Network'

        self._a_network = a_network
        self._b_network = b_network

        self._a_edge_fid = {}
        self._b_edge_fid = {}

        self._ab_node = {}
        self._ba_node = {}
        self._ab_edge = {}
        self._ba_edge = {}

        self._min_distance = kwargs.get('min_distance', 20)
        self._max_distance = kwargs.get('max_distance', 100)
        self._min_angle = kwargs.get('min_angle', 5)
        self._max_angle = kwargs.get('max_angle', 45)
        self._iterations = kwargs.get('iterations', 5)

        self._distance = self._min_distance
        self._angle = self._min_angle
        self._iteration = 1
        self._dirty = True

    def _networks(self):
        yield (self._a_network, self._b_network)
        yield (self._b_network, self._a_network)

    def _next_iteration(self):
        n = self._iterations - 1
        self._distance += float(self._max_distance - self._min_distance) / n
        self._angle += float(self._max_angle - self._min_angle) / n
        self._iteration += 1

    def _consolidate_eids(self, fid_map, eids):
        result = set()
        for eid in eids:
            if eid in fid_map:
                result |= self._consolidate_eids(fid_map, [eid])
                del fid_map[eid]
            else:
                result.add(eid)
        return result

    def _record_edge_merger(self, network, new_eid, old_eids):
        fid_map = self._a_edge_fid if network == self._a_network \
            else self._b_edge_fid
        fid_map[new_eid] = self._consolidate_eids(fid_map, old_eids)

    # Step 1
    def _remove_no_candidates(self):
        print('Removing no-candidate nodes and edges...')
        node_count = 0
        edge_count = 0

        for (network, other_network) in self._networks():
            for nid in list(network.nids()):
                node = network.get_node(nid)
                bbox = node.geometry().boundingBox().buffered(
                    self._max_distance)
                other_eids = other_network.find_eids(bbox)
                if not other_eids:
                    edge_count += len(network.get_node_eids(nid))
                    network.remove_node(nid)
                    node_count += 1

        if node_count > 0:
            self._dirty = True
        print('Removed %i nodes and %i edges' % (node_count, edge_count))

    # Step 2
    def _merge_continuous_edges(self):
        print('Merging continuous edges...')
        merge_count = 0
        for (network, other_network) in self._networks():
            for nid in list(network.nids()):
                eids = network.get_node_eids(nid)
                # Don't try to merge if...
                # More than two edges at this node
                if len(eids) != 2:
                    continue
                # Any of the edges are loops
                if True in [network.is_loop(eid) for eid in eids]:
                    continue
                # Both edges form a loop
                # TODO: How to deal with this? Remove them?
                nid_sets = [set(network.get_edge_nids(eid)) for eid in eids]
                if nid_sets[0] == nid_sets[1]:
                    continue
                new_eid = network.merge_edges(*eids)
                self._record_edge_merger(network, new_eid, eids)
                merge_count += 1

        if merge_count > 0:
            self._dirty = True
        print('Merged %i edge pairs' % (merge_count,))

    # Step 3
    def _match_nodes_to_nodes(self):
        print('Matching nodes to nodes...')
        pass

    # Step 4
    def _match_edges_to_edges(self):
        print('Matching edges to edges...')
        pass

    # Step 5
    def _match_nodes_to_edges(self):
        print('Matching nodes to edges...')
        pass

    def match(self):
        while self._iteration <= self._iterations:
            print('Starting iteration %i...' % (self._iteration))
            self._dirty = True

            while self._dirty:
                self._dirty = False
                self._remove_no_candidates()
                self._merge_continuous_edges()
                self._match_nodes_to_nodes()
                self._match_edges_to_edges()
                self._match_nodes_to_edges()

            self._next_iteration()


if __name__ == '__main__':
    qgs = QgsApplication([], False)
    qgs.initQgis()

    print('Creating networks...')
    time1 = time.time()
    a_net = Network('network.gpkg|layername=a_edge')
    b_net = Network('network.gpkg|layername=b_edge')
    time2 = time.time()
    print('Created networks in %0.3f sec' % (time2 - time1,))

    print('Matching...')
    matcher = Matcher(a_net, b_net)
    matcher.match()

    qgs.exitQgis()
