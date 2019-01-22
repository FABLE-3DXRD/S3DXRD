'''
This is a rough testing module for modelscanning3DXRD
The input and expected_output will need update whenever
there is a fundemental change to the physical model.
The idea is to run a forward model on the input
and check that the result is the same as a previous diffraction pattern.
This means that the expected_output patterns must bee reconstructed and
checked such that the user actually belive in their correctness.
If so has not been done these tests are usless.
These test are mainly used as a quick way of checking that everything is ok
with the code when minor changes has been introduced, like speed upgrades
or data structure changes.
'''

import unittest
from modelscanning3DXRD import gomodelscanning3DXRD
import os

class test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_one_orient_3x3(self):
        """
        Forward model a grain of 3x3 voxels.
        All voxels have the same orientation.
        All voxels have zero strain
        Assert that the contents of the .flt and .gve files ae correct
        """

        print_input=profile=killfile=debug=parallel=None
        input_file = "input/Sn_one_orient_3x3.inp"
        expected_flt_output = "expected_output/one_orient_3x3.flt"
        expected_gve_output = "expected_output/one_orient_3x3.gve"

        gomodelscanning3DXRD.run(print_input, input_file, killfile, profile, debug, parallel)
        output_flt = "output/one_orient_3x3_test.flt"
        output_gve = "output/one_orient_3x3_test.gve"

        self.assertTrue(self.compare(expected_flt_output,output_flt))
        self.assertTrue(self.compare(expected_gve_output,output_gve))

    def test_strain_3x3(self):
        """
        Forward model a grain of 3x3 voxels.
        All voxels have the same orientation.
        The grain carries a strain gradient in y. (strain in xx)
        Assert that the contents of the .flt and .gve files ae correct
        """

        print_input=profile=killfile=debug=parallel=None
        input_file = "input/Sn_strain_3x3.inp"
        expected_flt_output = "expected_output/strain_3x3.flt"
        expected_gve_output = "expected_output/strain_3x3.gve"

        gomodelscanning3DXRD.run(print_input, input_file, killfile, profile, debug, parallel)
        output_flt = "output/strain_3x3_test.flt"
        output_gve = "output/strain_3x3_test.gve"

        self.assertTrue(self.compare(expected_flt_output,output_flt))
        self.assertTrue(self.compare(expected_gve_output,output_gve))

    def test_three_grains_constant_strain_5x5(self):
        """
        Forward model a grain of 5x5 voxels.
        All voxels in each grain have the same orientation.
        All voxels in each grain have a uniform strain.
        Assert that the contents of the .flt and .gve files ae correct
        """

        print_input=profile=killfile=debug=parallel=None
        input_file = "input/Sn_three_grains_strain_5x5.inp"
        expected_flt_output = "expected_output/three_grains_strain_5x5.flt"
        expected_gve_output = "expected_output/three_grains_strain_5x5.gve"

        gomodelscanning3DXRD.run(print_input, input_file, killfile, profile, debug, parallel)
        output_flt = "output/three_grains_strain_5x5_test.flt"
        output_gve = "output/three_grains_strain_5x5_test.gve"

        self.assertTrue(self.compare(expected_flt_output,output_flt))
        self.assertTrue(self.compare(expected_gve_output,output_gve))


    def compare(self,file1,file2):
        """
        Compare two textfiles. Return True if they are identical.
        """

        with open(file1,'r') as f:
            string1 = (f.read()).lower().split("\n")

        if len(string1[0].split())>29:
            for i,line in enumerate(string1):
                l = line.split()
                try:
                    l.pop(29)
                except:
                    pass
                string1[i]="".join(l)

        with open(file2) as f:
            string2 = (f.read()).lower().split("\n")

        if len(string2[0].split())>29:
            for i,line in enumerate(string1):
                l = line.split()
                try:
                    l.pop(29)
                except:
                    pass
                string2[i]="".join(l)

        if len(string1)!=len(string2):
            return False

        missmatch=0
        for line1 in string1:
            if line1 not in string2:
                missmatch+=1

        if missmatch==0:
            return True
        else:
            return False


if __name__ == '__main__':
    os.chdir('../')
    os.system("python setup.py build")
    os.system("python setup.py install")
    os.chdir('test_voxelated/')

    unittest.main()

    os.remove('output/one_orient_3x3_test.flt')
    os.remove('output/one_orient_3x3_test.gve')
    os.remove('output/strain_3x3_test.flt')
    os.remove('output/strain_3x3_test.gve')
    os.remove('output/three_grains_strain_5x5_test.flt')
    os.remove('output/three_grains_strain_5x5_test.gve')











