# from qgis.core import QgsProject
from qgis.core import QgsApplication, QgsVectorLayer, QgsGeometry, \
    QgsSpatialIndex, QgsField, edit
from PyQt5.QtCore import QVariant


def average_distance(source_feature, target_feature):
    source = source_feature.geometry()
    target = target_feature.geometry()

    distances = [(i, target.distance(QgsGeometry(vertex)),
                 source.distanceToVertex(i))
                 for (i, vertex) in enumerate(source.vertices())]

    total = 0
    for (i, t, l) in distances[1:]:
        (pi, pt, pl) = distances[i - 1]
        total += (t + pt) * (l - pl) / 2

    return total / distances[-1][2]


class EdgeNetwork(object):

    def __init__(self, src):
        self._layer = QgsVectorLayer(src)
        self._map = {}
        self._index = QgsSpatialIndex()
        self._matches = {}

        for feature in self._layer.getFeatures():
            self._map[feature.id()] = feature
            self._index.insertFeature(feature)

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

    def neighbors(self, feature):
        return [n for n in self.near(feature, 0)
                if feature.geometry().touches(n.geometry())
                or n.id() == feature.id()]

    def is_matched(self, feature):
        return feature.id() in self._matches

    def match(self, self_feature, other_feature):
        if self_feature.id() not in self._matches:
            self._matches[self_feature.id()] = []

        self._matches[self_feature.id()].append(other_feature.id())

    def matches(self, self_feature, other_feature):
        return other_feature.id() in self._matches.get(self_feature.id(), [])

    def get_matches(self, feature):
        return self._matches.get(feature.id(), [])

    def neighbors_match(self, self_feature, other_feature, other_network):
        self_neighbors = self.neighbors(self_feature)
        other_neighbors = other_network.neighbors(other_feature)
        for self_neighbor in self_neighbors:
            for other_neighbor in other_neighbors:
                if self.matches(self_neighbor, other_neighbor):
                    return True
        return False

    def write_matches(self):
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
            for feature in self._layer.getFeatures():
                matches = self.get_matches(feature)
                feature_changes = {}
                feature_changes[matches_idx] = \
                    ', '.join([str(m) for m in matches])
                feature_changes[count_idx] = len(matches)
                changes[feature.id()] = feature_changes

            provider.changeAttributeValues(changes)


if __name__ == '__main__':
    qgs = QgsApplication([], False)
    qgs.initQgis()

    print('Creating networks...')
    a_net = EdgeNetwork('network.gpkg|layername=a_edge')
    b_net = EdgeNetwork('network.gpkg|layername=b_edge')

    print('Matching edges by similarity...')
    for a_edge in a_net.unmatched():
        for b_edge in b_net.near(a_edge, 25):
            if b_net.is_matched(b_edge):
                continue
            if a_edge.geometry().hausdorffDistance(b_edge.geometry()) <= 25:
                a_net.match(a_edge, b_edge)
                b_net.match(b_edge, a_edge)
                break

    print('Matching edges by connectivity...')

    rescan = True
    while rescan:
        print('Scanning unmatched...')
        rescan = False
        for a_edge in a_net.unmatched():
            for b_edge in b_net.near(a_edge, 25):
                if a_net.neighbors_match(a_edge, b_edge, b_net) and (
                        average_distance(a_edge, b_edge) <= 25 or
                        average_distance(b_edge, a_edge) <= 25):
                    a_net.match(a_edge, b_edge)
                    b_net.match(b_edge, a_edge)
                    rescan = True

    print('Writing matches...')
    a_net.write_matches()
    b_net.write_matches()

    qgs.exitQgis()
