"""Storing and writing data in RTTM format"""

from dataclasses import dataclass, field
import sys
import numpy as np

from click import FileError


@dataclass
class RttmObj:
    type: str
    file: str
    chnl: int
    tbeg: float
    tdur: float
    ortho: str = None
    stype: str = None
    name: str = None
    conf: float = None

    def check_attrs(self) -> None:
        if isinstance(self.tbeg, float) and self.tbeg < 0.0:
            raise ValueError("Attribute 'tbeg' must be greater than zero")
        if isinstance(self.tdur, float) and self.tdur < 0.0:
            raise ValueError("Attribute 'tdur' must be greater than zero")
        if isinstance(self.conf, float) and (self.conf < 0.0 or self.conf > 1.0):
            raise ValueError(
                "Attribute 'conf' must be greater than zero and smaller than 1.0")

    def __post_init__(self):
        self.check_attrs()


def get_default_header():
    return ["type", "file", "chnl", "tbeg", "tdur", "ortho", "stype", "name", "conf"]


def check_rttm_filename(filename):
    filename_split = filename.split(".")
    if filename_split[-1] != "rttm":
        raise FileError(filename, "Cannot open files without '.rttm' extension")


@dataclass
class RttmSeq:
    sequence: list[RttmObj]
    header: list[str] = field(default_factory=get_default_header)

    def __str__(self, end="\t", file=sys.stdout, header=True):
        if header:
            for h in self.header:
                print(h, end=end, file=file)

            print("", file=file)

        for obj in self.sequence:
            for _, value in obj.__dict__.items():
                if isinstance(value, type(None)):
                    print("<NA>", end=end, file=file)
                elif isinstance(value, float):
                    print(round(value, 2), end=end, file=file)
                else:
                    print(str(value), end=end, file=file)

            print("", file=file)

        return ""


    def write(self, filename):
        check_rttm_filename(filename)
        with open(filename, "w") as file:
            self.__str__(end=" ", file=file, header=False)


    def get_audio_segments(self, audio, sample_rate, max_length):
        speech_segments = []

        for seg in self.sequence:
            start = int(seg.tbeg*sample_rate)
            end = int(start + seg.tdur*sample_rate)
            length = end - start
            n = length // max_length + 1
            subsegments = np.array_split(audio[start:end], n)
            for subsegment in subsegments:
                speech_segments.append(subsegment)

        return speech_segments


def read_rttm(filename):
    check_rttm_filename(filename)
    with open(filename, "r") as file:
        sequence = []
        for row in file:
            row_split = [None if cell == "<NA>" else cell for cell in row.split(" ")]
            segment = RttmObj(
                type=row_split[0],
                file=row_split[1],
                chnl=int(row_split[2]),
                tbeg=float(row_split[3]),
                tdur=float(row_split[4]),
                ortho=row_split[5],
                stype=row_split[6],
                name=row_split[7],
                conf=float(row_split[8]) if not isinstance(row_split[8], type(None)) else None
            )
            sequence.append(segment)

    return RttmSeq(sequence, get_default_header())