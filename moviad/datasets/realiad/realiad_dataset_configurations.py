from enum import Enum

class RealIadDefectType(Enum):
    AK = "pit"
    BX = "deformation"
    CH = "abrasion"
    HS = "scratch"
    PS = "damage"
    QS = "missing parts"
    YW = "foreign objects"
    ZW = "contamination"


class RealIadAnomalyClass(Enum):
    OK = "OK"
    NG = "NG"
    AK = "AK"
    BX = "BX"
    CH = "CH"
    HS = "HS"
    PS = "PS"
    QS = "QS"
    YW = "YW"
    ZW = "ZW"

anomaly_class_encoding = {
    RealIadAnomalyClass.OK: 0,
    RealIadAnomalyClass.NG: 1,
    RealIadAnomalyClass.AK: 2,
    RealIadAnomalyClass.BX: 3,
    RealIadAnomalyClass.CH: 4,
    RealIadAnomalyClass.HS: 5,
    RealIadAnomalyClass.PS: 6,
    RealIadAnomalyClass.QS: 7,
    RealIadAnomalyClass.YW: 8,
    RealIadAnomalyClass.ZW: 9,
}


class RealIadClass(Enum):
    AUDIOJACK = "audiojack"
    BOTTLE = "bottle"
    CABLE = "cable"
    CAPSULE = "capsule"
    CARPET = "carpet"
    GRID = "grid"
    HAZELNUT = "hazelnut"
    LEATHER = "leather"
    METAL_NUT = "metal_nut"
    PILL = "pill"
    SCREW = "screw"
    TILE = "tile"
    TOOTHBRUSH = "toothbrush"
    TRANSISTOR = "transistor"
    WOOD = "wood"
    ZIPPER = "zipper"