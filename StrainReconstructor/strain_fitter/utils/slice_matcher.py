import numpy as np
import copy

class SliceMatcher(object):

    def __init__( self, index, label ):
        self.index = index 
        self.label = label
        self.ref_grain_shape = None
        self.ref_grain = None

    def set_reference(self, ref_grain_shape, ref_grain):
        if self.ref_grain==None:
            self.original_ref_grain = copy.deepcopy(ref_grain)
            self.original_ref_grain_shape = copy.deepcopy(ref_grain_shape)
        self.ref_grain_shape = ref_grain_shape
        self.ref_grain = ref_grain

    def reset_reference(self):
        self.ref_grain_shape = copy.deepcopy(self.original_ref_grain_shape)
        self.ref_grain = copy.deepcopy(self.original_ref_grain)

    def match_from_overlap( self, grain_shapes ):
        overlap = []
        for gs in grain_shapes:
            intersect = np.sum( self.ref_grain_shape*gs )
            total = np.sum( gs )
            overlap.append( intersect/float(total) )
        cond = np.max(np.array(overlap))/2.
        candidate_indxs = np.where( np.array(overlap)>cond )[0]
        return candidate_indxs

    def match_from_u( self, grains, candidate_indxs ):
        u_diffs = []
        for i in candidate_indxs:
            diff = self.ref_grain.u - grains[i].u
            u_diffs.append( np.linalg.norm( diff ) )
        indx = np.where( np.array(u_diffs)==np.min( u_diffs ) )[0]
        assert len(indx)==1
        indx = indx[0]
        return candidate_indxs[ indx ]
    
    def match( self, grains, grain_shapes ):
        candidate_indxs = self.match_from_overlap( grain_shapes )
        print("candidate_indxs ", candidate_indxs)
        if len(candidate_indxs)>0:
            return self.match_from_u( grains, candidate_indxs )
        else:
            print("No match in slice, end of grain !")
            return None