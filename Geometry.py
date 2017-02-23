class Point3D(object):

    def __init__(self, idx, pt3d, pt2d):
        self.idx = idx
        self.pt2d = pt2d
        self.pt3d = pt3d
        self.next = None
    def addNext(self, idx, pt3d, pt2d):
        self.next = Point3D(idx, pt3d, pt2d)
    
