"""Objects and routines for inverse kinematics in 2D space

"""
import numpy as np
try:
    from .linalg import *
except ValueError:
    from linalg import *


class CCD(object):
    """Cyclic coordinate decent algorithm

    """
    def __init__(self, linklengths):

        # array holding the joint angles and link lengths
        self._angles_linklengths = np.empty((len(linklengths), 2))
        self._angles_linklengths[:, 1] = np.array(linklengths)
        self.set_angles(len(linklengths)*[0.0])

        # Mappings from joint-space_{i+1} to join-space_i, for 0<=i<len(linklengths)
        self._backmappings = np.empty((len(linklengths), 3, 3))

        # Mappings from joint-space_0 to joint-space_i, for 0<=i<len(linklengths)
        self._forwardmappings = np.empty((len(linklengths), 3, 3))
        self._forwardmappings[0] = np.eye(3)


    def set_angles(self, angles):
        self._angles_linklengths[:, 0] = [degree_to_radian(a) for a in angles]
        for i in xrange(self._angles_linklengths.shape[0]):
            self._computebackmapping(i)
        self._computeforwardmappings()


    def get_angles(self):
        return [radian_to_degree(a) for a in self._angles_linklengths[:, 0]]


    def set_angle(self, i, angle):
        self._angles_linklengths[i, 0] = degree_to_radian(angle)
        self._computebackmapping(i)
        self._computeforwardmappings()


    def _computebackmapping(self, i):
        transrot_matrix(self._angles_linklengths[i, 0], self._angles_linklengths[i, 1], 0, out=self._backmappings[i])


    def _computeforwardmappings(self):
        for i, (angle, linklength) in enumerate(self._angles_linklengths[:-1, :]):
            np.dot(rottrans_matrix(-angle, -linklength, 0), self._forwardmappings[i], out=self._forwardmappings[i+1])


    def jointpositions(self):
        """Forward kinematics
        """
        accum = np.eye(3)
        for i in xrange(self._backmappings.shape[0]):
            accum = np.dot(accum, self._backmappings[i])
            p = np.dot(accum, (0, 0, 1))
            yield (p[0], p[1])


    def run(self, tx, ty):
        """Perform one iteration of the CCD algorithm

        Args:
            tx, ty (float): Target position

        Returns:
            float: Resulting distance between end-effector and (`tx`, `ty`)
        """
        ex_local, ey_local = 0, 0

        N = self._backmappings.shape[0] -1
        while N>=0:
            tx_local, ty_local, _ = np.dot(self._forwardmappings[N], (tx, ty, 1))
            ex_local1, ey_local1, _ = np.dot(self._backmappings[N], (ex_local, ey_local, 1))

            angle_et = angle((ex_local1, ey_local1), (tx_local, ty_local))

            self._angles_linklengths[N, 0] += angle_et
            self._computebackmapping(N)

            ex_local, ey_local, _ = np.dot(self._backmappings[N], (ex_local, ey_local, 1))

            N -= 1

        self._computeforwardmappings()

        import numpy.linalg as LA
        return LA.norm((ex_local - tx, ey_local - ty))



def test():
    ccd = CCD([1.0, 2.0, 1.0])
    ccd.set_angles([90, -90, 90])
    ccd._computeforwardmappings()

    r = ccd.run(1, 3)
    r = ccd.run(1, 3)
    r = ccd.run(1, 3)
    print "CCD", r, [radian_to_degree(a) for a in ccd.get_angles()]


if __name__=='__main__':
    test()