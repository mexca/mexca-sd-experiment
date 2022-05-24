from click import FileError
<<<<<<< Updated upstream
=======
from copy import copy, deepcopy
>>>>>>> Stashed changes
from rttm import RttmSeq, RttmObj, get_default_header, read_rttm
import pytest


def test_get_default_header():
    ref_header = ["type", "file", "chnl", "tbeg", "tdur", "ortho", "stype", "name", "conf"]
    default_header = get_default_header()

    for i, header in enumerate(default_header):
        assert header == ref_header[i]


default_header = get_default_header()


class TestRttmObj:
    def test_init(self):
        obj = RttmObj(
            type = "",
            file = "",
            chnl = 1,
            tbeg = 1.0,
            tdur = 1.0
        )

        for header in default_header:
            assert hasattr(obj, header)

    
    def test_check_attrs(self):
        with pytest.raises(ValueError):
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = -1.0,
                tdur = 1.0
            )

            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = -1.0
            )
            
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0,
                ortho = "",
                stype = "",
                name = "",
                conf = -0.4
            )

            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0,
                ortho = "",
                stype = "",
                name = "",
                conf = 1.4
            )

<<<<<<< Updated upstream
=======
    
    def test_copy(self):
        obj = RttmObj(
            type = "",
            file = "",
            chnl = 1,
            tbeg = 1.0,
            tdur = 1.0
        )
        obj_copy = copy(obj)

        assert obj is not obj_copy

>>>>>>> Stashed changes

class TestRttmSeq:
    def test_init(self):
        seq = [
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            ),
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            )
        ]
        obj = RttmSeq(seq)
        assert hasattr(obj, "sequence")


<<<<<<< Updated upstream
=======
    def test_copy(self):
        seq = [
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            ),
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            )
        ]
        obj = RttmSeq(seq)
        obj_copy = copy(obj)

        assert obj is not obj_copy


    def test_deepcopy(self):
        seq = [
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            ),
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            )
        ]
        obj = RttmSeq(seq)
        obj_copy = deepcopy(obj)

        assert obj.sequence[0] is not obj_copy.sequence[0]


    def test_sort(self):
        seq = [
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 2.0,
                tdur = 1.0
            ),
            RttmObj(
                type = "",
                file = "",
                chnl = 1,
                tbeg = 0.0,
                tdur = 1.0
            )
        ]
        obj = RttmSeq(seq)
        obj_sorted = obj.sort()
        
        assert obj_sorted.sequence[0].tbeg < obj_sorted.sequence[1].tbeg


>>>>>>> Stashed changes
    def test_read_write(self):
        test_filename = "test.rttm"
        seq = [
            RttmObj(
                type = "SEGMENT",
                file = "test",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            ),
            RttmObj(
                type = "SEGMENT",
                file = "test",
                chnl = 1,
                tbeg = 1.0,
                tdur = 1.0
            )
        ]
        obj = RttmSeq(seq)
        obj.write(test_filename)
        new_obj = read_rttm(test_filename)
        assert obj.__eq__(new_obj)

        with pytest.raises(FileError):
            obj.write("test.txt")
