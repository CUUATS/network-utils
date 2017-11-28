import time
from collections import defaultdict, namedtuple
from qgis.core import QgsApplication, QgsVectorLayer, QgsGeometry, \
    QgsSpatialIndex, QgsField, edit
from PyQt5.QtCore import QVariant


class EdgeNetwork(object):

    def __init__(self, src):
        self._layer = QgsVectorLayer(src)
        self._map = {}
        self._index = QgsSpatialIndex()
        self._node_edges = defaultdict(set)
        self._edge_nodes = {}
        self._matches = {}

        for feature in self._layer.getFeatures():
            self._map[feature.id()] = feature
            self._index.insertFeature(feature)

        for feature in self._layer.getFeatures():
            fid = feature.id()
            start = feature.geometry().constGet().startPoint()
            end = feature.geometry().constGet().endPoint()
            start_coords = (start.x(), start.y())
            end_coords = (end.x(), end.y())

            self._node_edges[start_coords].add(fid)
            self._node_edges[end_coords].add(fid)
            self._edge_nodes[fid] = [start_coords, end_coords]

    def _has_field(self, field_name):
        return field_name in [f.name() for f in self._layer.fields()]

    def _add_fields(self, fields):
        self._layer.dataProvider().addAttributes(
            [QgsField(fn, ft) for (fn, ft) in fields])

    def _iter(self, matched=False):
        self._layer.selectByIds(list(self._matches.keys()))
        if not matched:
            self._layer.invertSelection()

        for feature in self._layer.selectedFeatures():
            yield feature

    def get(self, feature_id):
        return self._map[feature_id]

    def matched(self):
        return self._iter(True)

    def unmatched(self):
        return self._iter()

    def near(self, feature, distance):
        bbox = feature.geometry().boundingBox().buffered(distance)
        return [self._map[fid] for fid in self._index.intersects(bbox)]

    def neighbors(self, fid):
        neighbors = set()
        for node in self._edge_nodes.get(fid, []):
            neighbors |= set(self._node_edges[node])
        neighbors.remove(fid)
        return neighbors

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


def max_distance(source_feature, target_feature):
    source = source_feature.geometry()
    target = target_feature.geometry()
    distance = 0
    for vertex in source.vertices():
        vertex_distance = QgsGeometry(vertex).distance(target)
        distance = max(distance, vertex_distance)
    return distance


def neighbors_mismatch(matches, self_neighbors, other_neighbors):
    for neighbor in self_neighbors:
        if matches[neighbor].isdisjoint(other_neighbors):
            return True
    return False


DISTANCE = 50
Candidate = namedtuple('Candidate', ['a', 'b', 'ab', 'ba'])

if __name__ == '__main__':
    qgs = QgsApplication([], False)
    qgs.initQgis()

    print('Creating networks...')
    time1 = time.time()
    a_net = EdgeNetwork('network.gpkg|layername=a_edge')
    b_net = EdgeNetwork('network.gpkg|layername=b_edge')
    time2 = time.time()
    print('Created networks in %0.3f sec' % (time2 - time1,))

    print('Finding candidates...')
    time1 = time.time()
    candidates = []
    for a in a_net.unmatched():
        for b in b_net.near(a, DISTANCE):
            ab = max_distance(a, b)
            ba = max_distance(b, a)
            if ab <= DISTANCE or ba <= DISTANCE:
                candidates.append(Candidate(a.id(), b.id(), ab, ba))
    time2 = time.time()
    print('Found candidates in %0.3f sec' % (time2 - time1,))

    print('Matching candidates...')
    time1 = time.time()
    candidates.sort(key=lambda c: -max(c.ab, c.ba))
    a_match = set()
    b_match = set()
    ab_match = defaultdict(set)
    ba_match = defaultdict(set)
    match_count = 0

    while candidates:
        candidate = candidates.pop()
        if candidate.a in a_match or candidate.b in b_match:
            continue
        a_neighbors = a_net.neighbors(candidate.a) & a_match
        b_neighbors = b_net.neighbors(candidate.b) & b_match
        if neighbors_mismatch(ab_match, a_neighbors, b_neighbors) or \
                neighbors_mismatch(ba_match, b_neighbors, a_neighbors):
            continue

        if candidate.ab < DISTANCE and candidate.ba < DISTANCE:
            print('Matched: %i (%0.1f) -> %i (%0.1f)' % (
                candidate.a, candidate.ab, candidate.b, candidate.ba))
            match_count += 1
            a_match.add(candidate.a)
            b_match.add(candidate.b)
            ab_match[candidate.a].add(candidate.b)
            ba_match[candidate.b].add(candidate.a)

    time2 = time.time()
    print('Matched %i candidates in %0.3f sec' % (match_count, time2 - time1,))

    print('Writing matches...')
    time1 = time.time()
    a_net.write_matches(ab_match)
    b_net.write_matches(ba_match)
    time2 = time.time()
    print('Wrote matches in %0.3f sec' % (time2 - time1,))

    qgs.exitQgis()
