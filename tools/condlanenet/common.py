COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (255, 0, 128),
    (0, 128, 255),
    (0, 255, 128),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (60, 180, 0),
    (180, 60, 0),
    (0, 60, 180),
    (0, 180, 60),
    (60, 0, 180),
    (180, 0, 60),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
]


def parse_lanes(preds, filename, img_shape):

    def normalize_coords(coords):
        res = []
        for coord in coords:
            res.append((int(coord[0] + 0.5), int(coord[1] + 0.5)))
        return res

    anno_dir = filename.replace('.jpg', '.lines.txt')
    preds = [normalize_coords(coord) for coord in preds]
    annos = []
    with open(anno_dir, 'r') as anno_f:
        lines = anno_f.readlines()
    for line in lines:
        coords = []
        numbers = line.strip().split(' ')
        coords_tmp = [float(n) for n in numbers]

        for i in range(len(coords_tmp) // 2):
            coords.append([coords_tmp[2 * i], coords_tmp[2 * i + 1]])
        annos.append(normalize_coords(coords))

    return preds, annos


def convert_coords_formal(lanes):
    res = []
    for lane in lanes:
        lane_coords = []
        for coord in lane:
            lane_coords.append({'x': coord[0], 'y': coord[1]})
        res.append(lane_coords)
    return res


def parse_anno(filename, formal=True):
    anno_dir = filename.replace('.jpg', '.lines.txt')
    annos = []
    with open(anno_dir, 'r') as anno_f:
        lines = anno_f.readlines()
    for line in lines:
        coords = []
        numbers = line.strip().split(' ')
        coords_tmp = [float(n) for n in numbers]

        for i in range(len(coords_tmp) // 2):
            coords.append((coords_tmp[2 * i], coords_tmp[2 * i + 1]))
        annos.append(coords)
    if formal:
        annos = convert_coords_formal(annos)
    return annos


def get_line_intersection(y, line, im_width):

    def in_line_range(val, start, end):
        s = min(start, end)
        e = max(start, end)
        if val >= s and val <= e and s != e:
            return True
        else:
            return False

    reg_x = -2
    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(y, point_start[1], point_end[1]):
            k = (point_end[0] - point_start[0]) / (
                point_end[1] - point_start[1])
            reg_x = int(k * (y - point_start[1]) + point_start[0] + 0.49999)
            break

    return reg_x


def tusimple_convert_formal(lanes, h_samples, im_width):
    lane_samples = []
    for lane in lanes:
        lane_sample = []
        for h in h_samples:
            x = get_line_intersection(h, lane, im_width)
            lane_sample.append(x)
        lane_samples.append(lane_sample)
    return lane_samples
