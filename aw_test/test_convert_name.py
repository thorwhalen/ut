from unittest import TestCase
from aw.reporting import _convert_name
from nose.tools import *
import aw.reporting as reporting
from aw.reporting import x_to_xml_name
from aw.reporting import x_to_disp_name
from aw.reporting import x_to_lu_name

__author__ = 'thorwhalen'


class TestConvertName(TestCase):
    def setUp(self):
        self.dict = {'a1': ['a2', 'a3'], 'b1': ['b2', 'b3']}

    def test_convert_name_with_list_returns_list_with_found(self):
        eq_(['a1', 'b1'], _convert_name(['a2', 'b2'], self.dict))
        eq_(['c2', 'b1'], _convert_name(['c2', 'b2'], self.dict))
        eq_(['r2', 'd2'], _convert_name(['r2', 'd2'], self.dict))
        eq_([], _convert_name([], self.dict))


    def test_mappings_of_actual_variables(self):
        lu_names = list(reporting.lu_name_x_dict.keys())
        xml_names = list(reporting.xml_x_dict.keys())
        disp_names = list(reporting.disp_x_dict.keys())
        eq_(x_to_disp_name(['clicks']),['Clicks'])
        eq_(x_to_lu_name(x_to_xml_name(x_to_disp_name(lu_names))),lu_names)
        eq_(x_to_disp_name(x_to_lu_name(x_to_xml_name(disp_names))),disp_names)
        eq_(x_to_xml_name(x_to_disp_name(x_to_lu_name(xml_names))),xml_names)