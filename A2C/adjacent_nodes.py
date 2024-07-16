adjacent_nodes_bo = {
    "230": ["231"],
    "231": ["230", "232"],
    "232": ["231", "233"],
    "233": ["232", "235"],
    "235": ["221", "233"],
    "221": ["210", "220", "235"],
    "220": ["209", "219", "221"],
    "219": ["218", "220", "282"],
    "218": ["219", "282"],
    "282": ["218", "219"],
    "273": [],
    "209": ["220"],
    "210": ["221"],
}

adjacent_nodes_3x3 = {
    "intersectionNW": ["intersectionN", "intersectionW"],
    "intersectionN": ["intersectionNE", "intersectionCenter", "intersectionNW"],
    "intersectionNE": ["intersectionN", "intersectionE"],
    "intersectionW": ["intersectionNW", "intersectionCenter", "intersectionSW"],
    "intersectionCenter": ["intersectionN", "intersectionS", "intersectionE", "intersectionW"],
    "intersectionE": ["intersectionNE", "intersectionCenter", "intersectionSE"],
    "intersectionSW": ["intersectionW", "intersectionS"],
    "intersectionS": ["intersectionSW", "intersectionCenter", "intersectionSE"],
    "intersectionSE": ["intersectionS", "intersectionE"]
}