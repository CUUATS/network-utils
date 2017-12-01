import math
import time
from collections import defaultdict
from qgis.core import QgsApplication, QgsVectorLayer, QgsGeometry, \
    QgsSpatialIndex, QgsField, edit, QgsFeature
from PyQt5.QtCore import QVariant


class Network(object):

    FIELDS = [
        ('matches', QVariant.String),
        ('matches_count', QVariant.Int),
        ('network_fid', QVariant.Int)]

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
        feature = self._make_feature(new_eid, geometry)

        self.remove_node(old_nid)
        self._edge_map[new_eid] = feature
        self._edge_index.insertFeature(feature)
        self._edge_nodes[new_eid] = endpoints
        for nid in endpoints:
            self._node_edges[nid].add(new_eid)
        return new_eid

    def write_matches(self, match_dict, network_fids):
        with edit(self._layer):
            new_fields = [(fn, ft) for (fn, ft) in self.FIELDS
                          if not self._has_field(fn)]
            if new_fields:
                self._add_fields(new_fields)
                self._layer.updateFields()

            provider = self._layer.dataProvider()
            matches_idx = provider.fieldNameIndex('matches')
            count_idx = provider.fieldNameIndex('matches_count')
            network_fid_idx = provider.fieldNameIndex('network_fid')

            changes = defaultdict(dict)
            for (fid, matches) in match_dict.items():
                matches = match_dict[fid]
                changes[fid][matches_idx] = \
                    ', '.join([str(m) for m in matches])
                changes[fid][count_idx] = len(matches)

            for (network_fid, fids) in network_fids.items():
                for fid in fids:
                    changes[fid][network_fid_idx] = network_fid

            provider.changeAttributeValues(dict(changes))


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

    def _choose(self, network, a, b):
        return a if network == self._a_network else b

    def _next_iteration(self):
        n = self._iterations - 1
        self._distance += float(self._max_distance - self._min_distance) / n
        self._angle += float(self._max_angle - self._min_angle) / n
        self._iteration += 1

    def _consolidate_eids(self, fid_map, eids):
        result = set()
        for eid in eids:
            if eid in fid_map:
                result |= self._consolidate_eids(fid_map, fid_map[eid])
                del fid_map[eid]
            else:
                result.add(eid)
        return result

    def _record_edge_merger(self, network, new_eid, old_eids):
        fid_map = self._choose(network, self._a_edge_fid, self._b_edge_fid)
        fid_map[new_eid] = self._consolidate_eids(fid_map, old_eids)

    def _get_unmatched_nids(self, network):
        node_matches = self._choose(network, self._ab_node, self._ba_node)
        return set(network.nids()) - set(node_matches.keys())

    def _angle_difference(self, a, b):
        diff = abs(a - b)
        return diff if diff <= 180 else 360 - diff

    def _angle_sets_match(self, a_angles, b_angles):
        if len(a_angles) != len(b_angles):
            return False

        matrix = [[self._angle_difference(a, b) <= self._angle
                  for b in b_angles] for a in a_angles]

        return all([True in r for r in matrix]) and \
            all([True in c for c in zip(*matrix)])

    def _nodes_can_match(self, a_nid, b_nid):
        a_node = self._a_network.get_node(a_nid)
        b_node = self._b_network.get_node(b_nid)

        if a_node.geometry().distance(b_node.geometry()) > self._distance:
            return False

        a_angles = self._a_network.get_node_angles(a_nid)
        b_angles = self._b_network.get_node_angles(b_nid)

        return self._angle_sets_match(a_angles.values(), b_angles.values())

    def _set_nodes_matched(self, a_nid, b_nid):
        self._ab_node[a_nid] = b_nid
        self._ba_node[b_nid] = a_nid

    def _set_edges_matched(self, a_eid, b_eid):
        self._ab_edge[a_eid] = b_eid
        self._ba_edge[b_eid] = a_eid
        self._a_network.remove_edge(a_eid)
        self._b_network.remove_edge(b_eid)

    # Step 1
    def _remove_no_candidates(self):
        print('\nRemoving no-candidate nodes and edges...')
        node_count = 0
        edge_count = 0

        for (network, other_network) in self._networks():
            for nid in self._get_unmatched_nids(network):
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
            for nid in self._get_unmatched_nids(network):
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
                # Merge the edges.
                old_eids = eids.copy()
                new_eid = network.merge_edges(*eids)
                self._record_edge_merger(network, new_eid, old_eids)
                merge_count += 1

        if merge_count > 0:
            self._dirty = True
        print('Merged %i edge pairs' % (merge_count,))

    # Step 3
    def _match_nodes_to_nodes(self):
        print('Matching nodes to nodes...')
        match_count = 0
        for a_nid in self._get_unmatched_nids(self._a_network):
            a_node = self._a_network.get_node(a_nid)
            bbox = a_node.geometry().boundingBox().buffered(self._distance)
            for b_nid in self._b_network.find_nids(bbox):
                if b_nid not in self._ba_node and \
                        self._nodes_can_match(a_nid, b_nid):
                    self._set_nodes_matched(a_nid, b_nid)
                    match_count += 1
                    break

        if match_count > 0:
            self._dirty = True
        print('Matched %i node pairs' % (match_count,))

    # Step 4
    def _match_edges_to_edges(self):
        print('Matching edges to edges...')
        match_count = 0
        for (a_nid, b_nid) in list(self._ab_node.items()):
            for a_eid in list(self._a_network.get_node_eids(a_nid)):
                a_other_nid = self._a_network.get_other_nid(a_eid, a_nid)
                expected_b_other_nid = self._ab_node.get(a_other_nid, None)
                if expected_b_other_nid is None:
                    continue
                for b_eid in list(self._b_network.get_node_eids(b_nid)):
                    b_other_nid = self._b_network.get_other_nid(b_eid, b_nid)
                    if b_other_nid != expected_b_other_nid:
                        continue
                    a_edge = self._a_network.get_edge(a_eid)
                    b_edge = self._b_network.get_edge(b_eid)
                    if a_edge.geometry().hausdorffDistance(b_edge.geometry()) \
                            <= self._distance:
                        self._set_edges_matched(a_eid, b_eid)
                        match_count += 1
                        break

        if match_count > 0:
            self._dirty = True
        print('Matched %i edge pairs' % (match_count,))

    # Step 5
    def _match_nodes_to_edges(self):
        print('Matching nodes to edges...')
        match_count = 0

        for (network, other_network) in self._networks():
            for nid in self._get_unmatched_nids(network):
                node = network.get_node(nid)
                bbox = node.geometry().boundingBox().buffered(
                    self._max_distance)
                eids = other_network.find_eids(bbox)
                for eid in eids:
                    node_geom = node.geometry()
                    edge = other_network.get_edge(eid)
                    edge_geom = edge.geometry()
                    if node_geom.distance(edge_geom) > self._distance:
                        continue

                    edge_node_loc = edge_geom.lineLocatePoint(node_geom)
                    edge_angle = edge_geom.interpolateAngle(
                        edge_node_loc) * 180 / math.pi
                    edge_angle_opp = edge_angle + 180 if edge_angle > 180 \
                        else edge_angle - 180
                    edge_angle_eid = None
                    edge_angle_opp_eid = None

                    node_angles = network.get_node_angles(nid)
                    for (node_eid, angle) in node_angles.items():
                        if self._angle_difference(edge_angle, angle) \
                                <= self._angle:
                            edge_angle_eid = node_eid
                        elif self._angle_difference(edge_angle_opp, angle) \
                                <= self._angle:
                            edge_angle_opp_eid = node_eid

                    if edge_angle_eid is not None \
                            and edge_angle_opp_eid is not None:

                        keep_eids = set([edge_angle_eid, edge_angle_opp_eid])
                        remove_eids = set(node_angles.keys()) - keep_eids
                        if remove_eids:
                            for remove_eid in remove_eids:
                                network.remove_edge(remove_eid)

                            match_count += 1
                            break

        if match_count > 0:
            self._dirty = True
        print('Matched %i nodes to edges' % (match_count,))

    def match(self):
        while self._iteration <= self._iterations:
            print('\n[Distance: %0.1f, Angle: %0.1f]' % (
                self._distance, self._angle))
            self._dirty = True

            while self._dirty:
                self._dirty = False
                self._remove_no_candidates()
                self._merge_continuous_edges()
                self._match_nodes_to_nodes()
                self._match_edges_to_edges()
                self._match_nodes_to_edges()

            self._next_iteration()

    def write_matches(self):
        ab = {}
        ba = {}

        for (a_eid, b_eid) in self._ab_edge.items():
            a_fids = self._a_edge_fid.get(a_eid, set([a_eid]))
            b_fids = self._b_edge_fid.get(b_eid, set([b_eid]))

            for a_fid in a_fids:
                ab[a_fid] = b_fids

            for b_fid in b_fids:
                ba[b_fid] = a_fids

        self._a_network.write_matches(ab, self._a_edge_fid)
        self._b_network.write_matches(ba, self._a_edge_fid)


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
    time1 = time.time()
    matcher = Matcher(a_net, b_net)
    matcher.match()
    time2 = time.time()
    print('Matched networks in %0.3f sec' % (time2 - time1,))

    print('Writing matches...')
    time1 = time.time()
    matcher.write_matches()
    time2 = time.time()
    print('Wrote matches in %0.3f sec' % (time2 - time1,))

    qgs.exitQgis()
