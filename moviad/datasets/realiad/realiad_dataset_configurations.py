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

class RealIadCategory:
    value: str
    json_path: str


REAL_IAD_CATEGORIES_JSONS = {
    "audiojack": "audiojack.json",
    "bottle": "bottle.json",
    "cable": "cable.json",
    "capsule": "capsule.json",
    "carpet": "carpet.json",
    "grid": "grid.json",
    "hazelnut": "hazelnut.json",
    "leather": "leather.json",
    "metal_nut": "metal_nut.json",
    "vcpill": "vcpill.json",
    "screw": "screw.json",
    "tile": "tile.json",
    "toothbrush": "toothbrush.json",
}

class RealIadClassEnum(Enum):
    TOY = "toy"
    PHONE_BATTERY = "phone_battery"
    AUDIOJACK = "audiojack"
    BOTTLE = "bottle"
    CABLE = "cable"
    PCB = "pcb"
    CAPSULE = "capsule"
    CARPET = "carpet"
    GRID = "grid"
    HAZELNUT = "hazelnut"
    LEATHER = "leather"
    METAL_NUT = "metal_nut"
    VCPILL = "vcpill"
    SCREW = "screw"
    TILE = "tile"
    TOOTHBRUSH = "toothbrush"
    TRANSISTOR = "transistor"
    WOOD = "wood"
    ZIPPER = "zipper"