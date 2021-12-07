# maybe worked with network_x
def get_maxx_maxy_non_pythonic(self, node_dict, nodes):
    maxx = -999999
    maxy = -999999
    for node, i in enumerate(nodes):
        # certainly a more pythonic way exists
        x = node_dict[i][0]
        if x > maxx:
            maxx = x
        y = node_dict[i][1]
        if y > maxy:
            maxy = y
    return maxx, maxy

