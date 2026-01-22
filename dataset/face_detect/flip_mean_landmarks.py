import numpy as np

# 加载原始平均人脸
path = '/data/xiaoshuai/facial_lanmark/train_1226/mean_face_20_d/mean_landmarks.npy'
mean_face_original = np.load(path)  # 形状: (235, 2)

dataset_info = dict(
    dataset_name='inshot',
    keypoint_info={
        0: dict(name='kpt-0', id=0, color=[255, 0, 0], type='', swap='kpt-36'),
        1: dict(name='kpt-1', id=1, color=[255, 0, 0], type='', swap='kpt-35'),
        2: dict(name='kpt-2', id=2, color=[255, 0, 0], type='', swap='kpt-34'),
        3: dict(name='kpt-3', id=3, color=[255, 0, 0], type='', swap='kpt-33'),
        4: dict(name='kpt-4', id=4, color=[255, 0, 0], type='', swap='kpt-32'),
        5: dict(name='kpt-5', id=5, color=[255, 0, 0], type='', swap='kpt-31'),
        6: dict(name='kpt-6', id=6, color=[255, 0, 0], type='', swap='kpt-30'),
        7: dict(name='kpt-7', id=7, color=[255, 0, 0], type='', swap='kpt-29'),
        8: dict(name='kpt-8', id=8, color=[255, 0, 0], type='', swap='kpt-28'),
        9: dict(name='kpt-9', id=9, color=[255, 0, 0], type='', swap='kpt-27'),
        10: dict(name='kpt-10', id=10, color=[255, 0, 0], type='', swap='kpt-26'),
        11: dict(name='kpt-11', id=11, color=[255, 0, 0], type='', swap='kpt-25'),
        12: dict(name='kpt-12', id=12, color=[255, 0, 0], type='', swap='kpt-24'),
        13: dict(name='kpt-13', id=13, color=[255, 0, 0], type='', swap='kpt-23'),
        14: dict(name='kpt-14', id=14, color=[255, 0, 0], type='', swap='kpt-22'),
        15: dict(name='kpt-15', id=15, color=[255, 0, 0], type='', swap='kpt-21'),
        16: dict(name='kpt-16', id=16, color=[255, 0, 0], type='', swap='kpt-20'),
        17: dict(name='kpt-17', id=17, color=[255, 0, 0], type='', swap='kpt-19'),
        18: dict(name='kpt-18', id=18, color=[255, 0, 0], type='', swap=''),
        19: dict(name='kpt-19', id=19, color=[255, 0, 0], type='', swap='kpt-17'),
        20: dict(name='kpt-20', id=20, color=[255, 0, 0], type='', swap='kpt-16'),
        21: dict(name='kpt-21', id=21, color=[255, 0, 0], type='', swap='kpt-15'),
        22: dict(name='kpt-22', id=22, color=[255, 0, 0], type='', swap='kpt-14'),
        23: dict(name='kpt-23', id=23, color=[255, 0, 0], type='', swap='kpt-13'),
        24: dict(name='kpt-24', id=24, color=[255, 0, 0], type='', swap='kpt-12'),
        25: dict(name='kpt-25', id=25, color=[255, 0, 0], type='', swap='kpt-11'),
        26: dict(name='kpt-26', id=26, color=[255, 0, 0], type='', swap='kpt-10'),
        27: dict(name='kpt-27', id=27, color=[255, 0, 0], type='', swap='kpt-9'),
        28: dict(name='kpt-28', id=28, color=[255, 0, 0], type='', swap='kpt-8'),
        29: dict(name='kpt-29', id=29, color=[255, 0, 0], type='', swap='kpt-7'),
        30: dict(name='kpt-30', id=30, color=[255, 0, 0], type='', swap='kpt-6'),
        31: dict(name='kpt-31', id=31, color=[255, 0, 0], type='', swap='kpt-5'),
        32: dict(name='kpt-32', id=32, color=[255, 0, 0], type='', swap='kpt-4'),
        33: dict(name='kpt-33', id=33, color=[255, 0, 0], type='', swap='kpt-3'),
        34: dict(name='kpt-34', id=34, color=[255, 0, 0], type='', swap='kpt-2'),
        35: dict(name='kpt-35', id=35, color=[255, 0, 0], type='', swap='kpt-1'),
        36: dict(name='kpt-36', id=36, color=[255, 0, 0], type='', swap='kpt-0'),
        37: dict(name='kpt-37', id=37, color=[255, 0, 0], type='', swap='kpt-67'),
        38: dict(name='kpt-38', id=38, color=[255, 0, 0], type='', swap='kpt-66'),
        39: dict(name='kpt-39', id=39, color=[255, 0, 0], type='', swap='kpt-65'),
        40: dict(name='kpt-40', id=40, color=[255, 0, 0], type='', swap='kpt-64'),
        41: dict(name='kpt-41', id=41, color=[255, 0, 0], type='', swap='kpt-63'),
        42: dict(name='kpt-42', id=42, color=[255, 0, 0], type='', swap='kpt-62'),
        43: dict(name='kpt-43', id=43, color=[255, 0, 0], type='', swap='kpt-61'),
        44: dict(name='kpt-44', id=44, color=[255, 0, 0], type='', swap='kpt-60'),
        45: dict(name='kpt-45', id=45, color=[255, 0, 0], type='', swap='kpt-59'),
        46: dict(name='kpt-46', id=46, color=[255, 0, 0], type='', swap='kpt-58'),
        47: dict(name='kpt-47', id=47, color=[255, 0, 0], type='', swap='kpt-57'),
        48: dict(name='kpt-48', id=48, color=[255, 0, 0], type='', swap='kpt-76'),
        49: dict(name='kpt-49', id=49, color=[255, 0, 0], type='', swap='kpt-75'),
        50: dict(name='kpt-50', id=50, color=[255, 0, 0], type='', swap='kpt-74'),
        51: dict(name='kpt-51', id=51, color=[255, 0, 0], type='', swap='kpt-73'),
        52: dict(name='kpt-52', id=52, color=[255, 0, 0], type='', swap='kpt-72'),
        53: dict(name='kpt-53', id=53, color=[255, 0, 0], type='', swap='kpt-71'),
        54: dict(name='kpt-54', id=54, color=[255, 0, 0], type='', swap='kpt-70'),
        55: dict(name='kpt-55', id=55, color=[255, 0, 0], type='', swap='kpt-69'),
        56: dict(name='kpt-56', id=56, color=[255, 0, 0], type='', swap='kpt-68'),
        57: dict(name='kpt-57', id=57, color=[255, 0, 0], type='', swap='kpt-47'),
        58: dict(name='kpt-58', id=58, color=[255, 0, 0], type='', swap='kpt-46'),
        59: dict(name='kpt-59', id=59, color=[255, 0, 0], type='', swap='kpt-45'),
        60: dict(name='kpt-60', id=60, color=[255, 0, 0], type='', swap='kpt-44'),
        61: dict(name='kpt-61', id=61, color=[255, 0, 0], type='', swap='kpt-43'),
        62: dict(name='kpt-62', id=62, color=[255, 0, 0], type='', swap='kpt-42'),
        63: dict(name='kpt-63', id=63, color=[255, 0, 0], type='', swap='kpt-41'),
        64: dict(name='kpt-64', id=64, color=[255, 0, 0], type='', swap='kpt-40'),
        65: dict(name='kpt-65', id=65, color=[255, 0, 0], type='', swap='kpt-39'),
        66: dict(name='kpt-66', id=66, color=[255, 0, 0], type='', swap='kpt-38'),
        67: dict(name='kpt-67', id=67, color=[255, 0, 0], type='', swap='kpt-37'),
        68: dict(name='kpt-68', id=68, color=[255, 0, 0], type='', swap='kpt-56'),
        69: dict(name='kpt-69', id=69, color=[255, 0, 0], type='', swap='kpt-55'),
        70: dict(name='kpt-70', id=70, color=[255, 0, 0], type='', swap='kpt-54'),
        71: dict(name='kpt-71', id=71, color=[255, 0, 0], type='', swap='kpt-53'),
        72: dict(name='kpt-72', id=72, color=[255, 0, 0], type='', swap='kpt-52'),
        73: dict(name='kpt-73', id=73, color=[255, 0, 0], type='', swap='kpt-51'),
        74: dict(name='kpt-74', id=74, color=[255, 0, 0], type='', swap='kpt-50'),
        75: dict(name='kpt-75', id=75, color=[255, 0, 0], type='', swap='kpt-49'),
        76: dict(name='kpt-76', id=76, color=[255, 0, 0], type='', swap='kpt-48'),
        # 添加77-100点 (左眼轮廓)
        77: dict(name='kpt-77', id=77, color=[255, 0, 0], type='', swap='kpt-113'),
        78: dict(name='kpt-78', id=78, color=[255, 0, 0], type='', swap='kpt-112'),
        79: dict(name='kpt-79', id=79, color=[255, 0, 0], type='', swap='kpt-111'),
        80: dict(name='kpt-80', id=80, color=[255, 0, 0], type='', swap='kpt-110'),
        81: dict(name='kpt-81', id=81, color=[255, 0, 0], type='', swap='kpt-109'),
        82: dict(name='kpt-82', id=82, color=[255, 0, 0], type='', swap='kpt-108'),
        83: dict(name='kpt-83', id=83, color=[255, 0, 0], type='', swap='kpt-107'),
        84: dict(name='kpt-84', id=84, color=[255, 0, 0], type='', swap='kpt-106'),
        85: dict(name='kpt-85', id=85, color=[255, 0, 0], type='', swap='kpt-105'),
        86: dict(name='kpt-86', id=86, color=[255, 0, 0], type='', swap='kpt-104'),
        87: dict(name='kpt-87', id=87, color=[255, 0, 0], type='', swap='kpt-103'),
        88: dict(name='kpt-88', id=88, color=[255, 0, 0], type='', swap='kpt-102'),
        89: dict(name='kpt-89', id=89, color=[255, 0, 0], type='', swap='kpt-101'),
        90: dict(name='kpt-90', id=90, color=[255, 0, 0], type='', swap='kpt-124'),
        91: dict(name='kpt-91', id=91, color=[255, 0, 0], type='', swap='kpt-123'),
        92: dict(name='kpt-92', id=92, color=[255, 0, 0], type='', swap='kpt-122'),
        93: dict(name='kpt-93', id=93, color=[255, 0, 0], type='', swap='kpt-121'),
        94: dict(name='kpt-94', id=94, color=[255, 0, 0], type='', swap='kpt-120'),
        95: dict(name='kpt-95', id=95, color=[255, 0, 0], type='', swap='kpt-119'),
        96: dict(name='kpt-96', id=96, color=[255, 0, 0], type='', swap='kpt-118'),
        97: dict(name='kpt-97', id=97, color=[255, 0, 0], type='', swap='kpt-117'),
        98: dict(name='kpt-98', id=98, color=[255, 0, 0], type='', swap='kpt-116'),
        99: dict(name='kpt-99', id=99, color=[255, 0, 0], type='', swap='kpt-115'),
        100: dict(name='kpt-100', id=100, color=[255, 0, 0], type='', swap='kpt-114'),
        # 添加101-124点 (右眼轮廓)
        101: dict(name='kpt-101', id=101, color=[255, 0, 0], type='', swap='kpt-89'),
        102: dict(name='kpt-102', id=102, color=[255, 0, 0], type='', swap='kpt-88'),
        103: dict(name='kpt-103', id=103, color=[255, 0, 0], type='', swap='kpt-87'),
        104: dict(name='kpt-104', id=104, color=[255, 0, 0], type='', swap='kpt-86'),
        105: dict(name='kpt-105', id=105, color=[255, 0, 0], type='', swap='kpt-85'),
        106: dict(name='kpt-106', id=106, color=[255, 0, 0], type='', swap='kpt-84'),
        107: dict(name='kpt-107', id=107, color=[255, 0, 0], type='', swap='kpt-83'),
        108: dict(name='kpt-108', id=108, color=[255, 0, 0], type='', swap='kpt-82'),
        109: dict(name='kpt-109', id=109, color=[255, 0, 0], type='', swap='kpt-81'),
        110: dict(name='kpt-110', id=110, color=[255, 0, 0], type='', swap='kpt-80'),
        111: dict(name='kpt-111', id=111, color=[255, 0, 0], type='', swap='kpt-79'),
        112: dict(name='kpt-112', id=112, color=[255, 0, 0], type='', swap='kpt-78'),
        113: dict(name='kpt-113', id=113, color=[255, 0, 0], type='', swap='kpt-77'),
        114: dict(name='kpt-114', id=114, color=[255, 0, 0], type='', swap='kpt-100'),
        115: dict(name='kpt-115', id=115, color=[255, 0, 0], type='', swap='kpt-99'),
        116: dict(name='kpt-116', id=116, color=[255, 0, 0], type='', swap='kpt-98'),
        117: dict(name='kpt-117', id=117, color=[255, 0, 0], type='', swap='kpt-97'),
        118: dict(name='kpt-118', id=118, color=[255, 0, 0], type='', swap='kpt-96'),
        119: dict(name='kpt-119', id=119, color=[255, 0, 0], type='', swap='kpt-95'),
        120: dict(name='kpt-120', id=120, color=[255, 0, 0], type='', swap='kpt-94'),
        121: dict(name='kpt-121', id=121, color=[255, 0, 0], type='', swap='kpt-93'),
        122: dict(name='kpt-122', id=122, color=[255, 0, 0], type='', swap='kpt-92'),
        123: dict(name='kpt-123', id=123, color=[255, 0, 0], type='', swap='kpt-91'),
        124: dict(name='kpt-124', id=124, color=[255, 0, 0], type='', swap='kpt-90'),
        # 添加125-140点 (鼻子区域)
        125: dict(name='kpt-125', id=125, color=[255, 0, 0], type='', swap='kpt-136'),
        126: dict(name='kpt-126', id=126, color=[255, 0, 0], type='', swap='kpt-135'),
        127: dict(name='kpt-127', id=127, color=[255, 0, 0], type='', swap='kpt-134'),
        128: dict(name='kpt-128', id=128, color=[255, 0, 0], type='', swap='kpt-133'),
        129: dict(name='kpt-129', id=129, color=[255, 0, 0], type='', swap='kpt-132'),
        130: dict(name='kpt-130', id=130, color=[255, 0, 0], type='', swap='kpt-131'),
        131: dict(name='kpt-131', id=131, color=[255, 0, 0], type='', swap='kpt-130'),
        132: dict(name='kpt-132', id=132, color=[255, 0, 0], type='', swap='kpt-129'),
        133: dict(name='kpt-133', id=133, color=[255, 0, 0], type='', swap='kpt-128'),
        134: dict(name='kpt-134', id=134, color=[255, 0, 0], type='', swap='kpt-127'),
        135: dict(name='kpt-135', id=135, color=[255, 0, 0], type='', swap='kpt-126'),
        136: dict(name='kpt-136', id=136, color=[255, 0, 0], type='', swap='kpt-125'),
        137: dict(name='kpt-137', id=137, color=[255, 0, 0], type='', swap=''),
        138: dict(name='kpt-138', id=138, color=[255, 0, 0], type='', swap=''),
        139: dict(name='kpt-139', id=139, color=[255, 0, 0], type='', swap=''),
        140: dict(name='kpt-140', id=140, color=[255, 0, 0], type='', swap=''),
        # 添加141-200点 (嘴巴区域)
        141: dict(name='kpt-141', id=141, color=[255, 0, 0], type='', swap='kpt-159'),
        142: dict(name='kpt-142', id=142, color=[255, 0, 0], type='', swap='kpt-158'),
        143: dict(name='kpt-143', id=143, color=[255, 0, 0], type='', swap='kpt-157'),
        144: dict(name='kpt-144', id=144, color=[255, 0, 0], type='', swap='kpt-156'),
        145: dict(name='kpt-145', id=145, color=[255, 0, 0], type='', swap='kpt-155'),
        146: dict(name='kpt-146', id=146, color=[255, 0, 0], type='', swap='kpt-154'),
        147: dict(name='kpt-147', id=147, color=[255, 0, 0], type='', swap='kpt-153'),
        148: dict(name='kpt-148', id=148, color=[255, 0, 0], type='', swap='kpt-152'),
        149: dict(name='kpt-149', id=149, color=[255, 0, 0], type='', swap='kpt-151'),
        150: dict(name='kpt-150', id=150, color=[255, 0, 0], type='', swap=''),
        151: dict(name='kpt-151', id=151, color=[255, 0, 0], type='', swap='kpt-149'),
        152: dict(name='kpt-152', id=152, color=[255, 0, 0], type='', swap='kpt-148'),
        153: dict(name='kpt-153', id=153, color=[255, 0, 0], type='', swap='kpt-147'),
        154: dict(name='kpt-154', id=154, color=[255, 0, 0], type='', swap='kpt-146'),
        155: dict(name='kpt-155', id=155, color=[255, 0, 0], type='', swap='kpt-145'),
        156: dict(name='kpt-156', id=156, color=[255, 0, 0], type='', swap='kpt-144'),
        157: dict(name='kpt-157', id=157, color=[255, 0, 0], type='', swap='kpt-143'),
        158: dict(name='kpt-158', id=158, color=[255, 0, 0], type='', swap='kpt-142'),
        159: dict(name='kpt-159', id=159, color=[255, 0, 0], type='', swap='kpt-141'),
        160: dict(name='kpt-160', id=160, color=[255, 0, 0], type='', swap='kpt-176'),
        161: dict(name='kpt-161', id=161, color=[255, 0, 0], type='', swap='kpt-175'),
        162: dict(name='kpt-162', id=162, color=[255, 0, 0], type='', swap='kpt-174'),
        163: dict(name='kpt-163', id=163, color=[255, 0, 0], type='', swap='kpt-173'),
        164: dict(name='kpt-164', id=164, color=[255, 0, 0], type='', swap='kpt-172'),
        165: dict(name='kpt-165', id=165, color=[255, 0, 0], type='', swap='kpt-171'),
        166: dict(name='kpt-166', id=166, color=[255, 0, 0], type='', swap='kpt-170'),
        167: dict(name='kpt-167', id=167, color=[255, 0, 0], type='', swap='kpt-169'),
        168: dict(name='kpt-168', id=168, color=[255, 0, 0], type='', swap=''),
        169: dict(name='kpt-169', id=169, color=[255, 0, 0], type='', swap='kpt-167'),
        170: dict(name='kpt-170', id=170, color=[255, 0, 0], type='', swap='kpt-166'),
        171: dict(name='kpt-171', id=171, color=[255, 0, 0], type='', swap='kpt-165'),
        172: dict(name='kpt-172', id=172, color=[255, 0, 0], type='', swap='kpt-164'),
        173: dict(name='kpt-173', id=173, color=[255, 0, 0], type='', swap='kpt-163'),
        174: dict(name='kpt-174', id=174, color=[255, 0, 0], type='', swap='kpt-162'),
        175: dict(name='kpt-175', id=175, color=[255, 0, 0], type='', swap='kpt-161'),
        176: dict(name='kpt-176', id=176, color=[255, 0, 0], type='', swap='kpt-160'),
        177: dict(name='kpt-177', id=177, color=[255, 0, 0], type='', swap='kpt-189'),
        178: dict(name='kpt-178', id=178, color=[255, 0, 0], type='', swap='kpt-188'),
        179: dict(name='kpt-179', id=179, color=[255, 0, 0], type='', swap='kpt-187'),
        180: dict(name='kpt-180', id=180, color=[255, 0, 0], type='', swap='kpt-186'),
        181: dict(name='kpt-181', id=181, color=[255, 0, 0], type='', swap='kpt-185'),
        182: dict(name='kpt-182', id=182, color=[255, 0, 0], type='', swap='kpt-184'),
        183: dict(name='kpt-183', id=183, color=[255, 0, 0], type='', swap=''),
        184: dict(name='kpt-184', id=184, color=[255, 0, 0], type='', swap='kpt-182'),
        185: dict(name='kpt-185', id=185, color=[255, 0, 0], type='', swap='kpt-181'),
        186: dict(name='kpt-186', id=186, color=[255, 0, 0], type='', swap='kpt-180'),
        187: dict(name='kpt-187', id=187, color=[255, 0, 0], type='', swap='kpt-179'),
        188: dict(name='kpt-188', id=188, color=[255, 0, 0], type='', swap='kpt-178'),
        189: dict(name='kpt-189', id=189, color=[255, 0, 0], type='', swap='kpt-177'),
        190: dict(name='kpt-190', id=190, color=[255, 0, 0], type='', swap='kpt-200'),
        191: dict(name='kpt-191', id=191, color=[255, 0, 0], type='', swap='kpt-199'),
        192: dict(name='kpt-192', id=192, color=[255, 0, 0], type='', swap='kpt-198'),
        193: dict(name='kpt-193', id=193, color=[255, 0, 0], type='', swap='kpt-197'),
        194: dict(name='kpt-194', id=194, color=[255, 0, 0], type='', swap='kpt-196'),
        195: dict(name='kpt-195', id=195, color=[255, 0, 0], type='', swap=''),
        196: dict(name='kpt-196', id=196, color=[255, 0, 0], type='', swap='kpt-194'),
        197: dict(name='kpt-197', id=197, color=[255, 0, 0], type='', swap='kpt-193'),
        198: dict(name='kpt-198', id=198, color=[255, 0, 0], type='', swap='kpt-192'),
        199: dict(name='kpt-199', id=199, color=[255, 0, 0], type='', swap='kpt-191'),
        200: dict(name='kpt-200', id=200, color=[255, 0, 0], type='', swap='kpt-190'),
        # 添加201-202点 (瞳孔)
        201: dict(name='kpt-201', id=201, color=[255, 0, 0], type='', swap='kpt-202'),
        202: dict(name='kpt-202', id=202, color=[255, 0, 0], type='', swap='kpt-201'),
        # 添加203-218点 (左眼眼球轮廓)
        203: dict(name='kpt-203', id=203, color=[255, 0, 0], type='', swap='kpt-219'),
        204: dict(name='kpt-204', id=204, color=[255, 0, 0], type='', swap='kpt-234'),
        205: dict(name='kpt-205', id=205, color=[255, 0, 0], type='', swap='kpt-233'),
        206: dict(name='kpt-206', id=206, color=[255, 0, 0], type='', swap='kpt-232'),
        207: dict(name='kpt-207', id=207, color=[255, 0, 0], type='', swap='kpt-231'),
        208: dict(name='kpt-208', id=208, color=[255, 0, 0], type='', swap='kpt-230'),
        209: dict(name='kpt-209', id=209, color=[255, 0, 0], type='', swap='kpt-229'),
        210: dict(name='kpt-210', id=210, color=[255, 0, 0], type='', swap='kpt-228'),
        211: dict(name='kpt-211', id=211, color=[255, 0, 0], type='', swap='kpt-227'),
        212: dict(name='kpt-212', id=212, color=[255, 0, 0], type='', swap='kpt-226'),
        213: dict(name='kpt-213', id=213, color=[255, 0, 0], type='', swap='kpt-225'),
        214: dict(name='kpt-214', id=214, color=[255, 0, 0], type='', swap='kpt-224'),
        215: dict(name='kpt-215', id=215, color=[255, 0, 0], type='', swap='kpt-223'),
        216: dict(name='kpt-216', id=216, color=[255, 0, 0], type='', swap='kpt-222'),
        217: dict(name='kpt-217', id=217, color=[255, 0, 0], type='', swap='kpt-221'),
        218: dict(name='kpt-218', id=218, color=[255, 0, 0], type='', swap='kpt-220'),
        # 添加219-234点 (右眼眼球轮廓)
        219: dict(name='kpt-219', id=219, color=[255, 0, 0], type='', swap='kpt-203'),
        220: dict(name='kpt-220', id=220, color=[255, 0, 0], type='', swap='kpt-218'),
        221: dict(name='kpt-221', id=221, color=[255, 0, 0], type='', swap='kpt-217'),
        222: dict(name='kpt-222', id=222, color=[255, 0, 0], type='', swap='kpt-216'),
        223: dict(name='kpt-223', id=223, color=[255, 0, 0], type='', swap='kpt-215'),
        224: dict(name='kpt-224', id=224, color=[255, 0, 0], type='', swap='kpt-214'),
        225: dict(name='kpt-225', id=225, color=[255, 0, 0], type='', swap='kpt-213'),
        226: dict(name='kpt-226', id=226, color=[255, 0, 0], type='', swap='kpt-212'),
        227: dict(name='kpt-227', id=227, color=[255, 0, 0], type='', swap='kpt-211'),
        228: dict(name='kpt-228', id=228, color=[255, 0, 0], type='', swap='kpt-210'),
        229: dict(name='kpt-229', id=229, color=[255, 0, 0], type='', swap='kpt-209'),
        230: dict(name='kpt-230', id=230, color=[255, 0, 0], type='', swap='kpt-208'),
        231: dict(name='kpt-231', id=231, color=[255, 0, 0], type='', swap='kpt-207'),
        232: dict(name='kpt-232', id=232, color=[255, 0, 0], type='', swap='kpt-206'),
        233: dict(name='kpt-233', id=233, color=[255, 0, 0], type='', swap='kpt-205'),
        234: dict(name='kpt-234', id=234, color=[255, 0, 0], type='', swap='kpt-204'),
    })

def build_swap_mapping(dataset_info, num_points=235):
    # 初始化swap映射字典
    swap_mapping = {}
    
    # 首先，根据提供的映射建立初始映射
    for idx in range(num_points):
        # 如果这个点在dataset_info中
        if idx in dataset_info['keypoint_info']:
            swap_str = dataset_info['keypoint_info'][idx]['swap']
            if swap_str and swap_str != '':
                # 解析swap字符串，例如'kpt-36' -> 36
                swap_idx = int(swap_str.split('-')[-1])
                swap_mapping[idx] = swap_idx
            else:
                # 没有swap的点，映射到自身
                swap_mapping[idx] = idx
    
    return swap_mapping

def flip_landmarks_with_swap(landmarks, swap_mapping, image_width=1.0):
    num_points = landmarks.shape[0]
    flipped = np.zeros_like(landmarks)
    
    for i in range(num_points):
        # 获取这个点翻转后对应的索引
        swap_idx = swap_mapping.get(i, i)  # 如果没有映射，则映射到自身
        
        # 如果swap_idx超出了landmarks的范围，则使用自身
        if swap_idx >= num_points:
            swap_idx = i
        
        # 获取原始点的坐标
        original_point = landmarks[swap_idx]
        
        # x坐标翻转: x' = width - x
        flipped[i, 0] = image_width - original_point[0]
        # y坐标不变
        flipped[i, 1] = original_point[1]
    
    return flipped

swap_mapping = build_swap_mapping(dataset_info, num_points=235)

mean_face_flipped = flip_landmarks_with_swap(mean_face_original, swap_mapping)

mean_face_symmetric = (mean_face_original + mean_face_flipped) / 2

np.save('mean_face_original.npy', mean_face_original)
np.save('mean_face_flipped.npy', mean_face_flipped)
np.save('mean_face_symmetric.npy', mean_face_symmetric)


image_width = 1.0
image_height = 1.0
image_center_x = 0.5
image_center_y = 0.5

# 计算均值人脸的边界框
x_coords = mean_face_symmetric[:, 0]
y_coords = mean_face_symmetric[:, 1]

min_x, max_x = np.min(x_coords), np.max(x_coords)
min_y, max_y = np.min(y_coords), np.max(y_coords)

# 计算边界框中心
bbox_center_x = (min_x + max_x) / 2
bbox_center_y = (min_y + max_y) / 2

# 计算需要平移的偏移量
offset_x = image_center_x - bbox_center_x
offset_y = image_center_y - bbox_center_y

# 平移均值人脸，使其边界框中心与图像中心重合
centered_mean_face = mean_face_symmetric.copy()
centered_mean_face[:, 0] += offset_x
centered_mean_face[:, 1] += offset_y

# 重新计算平移后的边界框
centered_x_coords = centered_mean_face[:, 0]
centered_y_coords = centered_mean_face[:, 1]

centered_min_x, centered_max_x = np.min(centered_x_coords), np.max(centered_x_coords)
centered_min_y, centered_max_y = np.min(centered_y_coords), np.max(centered_y_coords)

centered_bbox_center_x = (centered_min_x + centered_max_x) / 2
centered_bbox_center_y = (centered_min_y + centered_max_y) / 2

# 计算正方形框（以最大边长为边长）
square_size = max(centered_max_x - centered_min_x, centered_max_y - centered_min_y)

# 计算正方形框的位置（使其中心也在图像中心）
square_min_x = image_center_x - square_size / 2
square_max_x = image_center_x + square_size / 2
square_min_y = image_center_y - square_size / 2
square_max_y = image_center_y + square_size / 2

square_box_pixel = np.array([square_min_x, square_min_y, square_max_x, square_max_y])

np.save('mean_face_symmetric_centered.npy', centered_mean_face)

print("=== 原始均值人脸 ===")
print(f"边界框: x[{min_x:.4f}, {max_x:.4f}], y[{min_y:.4f}, {max_y:.4f}]")
print(f"边界框中心: ({bbox_center_x:.4f}, {bbox_center_y:.4f})")
print(f"图像中心: ({image_center_x:.4f}, {image_center_y:.4f})")
print(f"偏移量: x={offset_x:.4f}, y={offset_y:.4f}")

print("\n=== 中心化后的均值人脸 ===")
print(f"边界框: x[{centered_min_x:.4f}, {centered_max_x:.4f}], y[{centered_min_y:.4f}, {centered_max_y:.4f}]")
print(f"边界框中心: ({centered_bbox_center_x:.4f}, {centered_bbox_center_y:.4f})")

print("\n=== 外包正方形框 ===")
print(f"正方形框: 左上({square_min_x:.4f}, {square_min_y:.4f}), 右下({square_max_x:.4f}, {square_max_y:.4f})")
print(f"正方形框中心: ({image_center_x:.4f}, {image_center_y:.4f})")
print(f"正方形边长: {square_size:.4f}")


try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 归一化坐标可视化
    axes[0].scatter(centered_mean_face[:, 0], centered_mean_face[:, 1], 
                    c='blue', s=10, alpha=0.7, label='Centered Landmarks')
    
    # 绘制正方形框
    from matplotlib.patches import Rectangle
    rect = Rectangle((square_min_x, square_min_y), square_size, square_size,
                     fill=False, edgecolor='red', linewidth=2, label='Square Box')
    axes[0].add_patch(rect)
    
    # 标记中心点
    axes[0].scatter([image_center_x], [image_center_y], c='green', s=100, 
                    marker='+', linewidths=2, label='Image Center')
    
    axes[0].set_xlabel('X (normalized)')
    axes[0].set_ylabel('Y (normalized)')
    axes[0].set_title('Centered Mean Face with Square Box (Normalized)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()  # 图像坐标系Y轴向下
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1, 0)
    
    # 像素坐标可视化
    axes[1].scatter(centered_mean_face[:, 0], centered_mean_face[:, 1], 
                    c='blue', s=10, alpha=0.7, label='Centered Landmarks')
    
    # 绘制正方形框（像素）
    rect_pixel = Rectangle((square_box_pixel[0], square_box_pixel[1]), 
                          square_box_pixel[2] - square_box_pixel[0],
                          square_box_pixel[3] - square_box_pixel[1],
                          fill=False, edgecolor='red', linewidth=2, label='Square Box')
    axes[1].add_patch(rect_pixel)
    
    # 标记中心点（像素）
    axes[1].scatter([96/2], [96/2], c='green', s=100, 
                    marker='+', linewidths=2, label='Image Center')
    
    axes[1].set_xlabel('X (pixels)')
    axes[1].set_ylabel('Y (pixels)')
    axes[1].set_title('Centered Mean Face with Square Box (96x96 Pixels)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 96)
    axes[1].set_ylim(96, 0)
    
    plt.tight_layout()
    plt.savefig('mean_face_square_box_visualization.png')
    
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始平均人脸
    axes[0].scatter(mean_face_original[:, 0], mean_face_original[:, 1], 
                    c='blue', s=5, alpha=0.7)
    axes[0].set_title('Original Mean Face')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(1, 0)
    
    # Flip后的人脸
    axes[1].scatter(mean_face_flipped[:, 0], mean_face_flipped[:, 1], 
                    c='red', s=5, alpha=0.7)
    axes[1].set_title('Flipped Face (with swap mapping)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1, 0)
    
    # 对称平均人脸
    axes[2].scatter(mean_face_symmetric[:, 0], mean_face_symmetric[:, 1], 
                    c='green', s=5, alpha=0.7)
    axes[2].set_title('Symmetric Mean Face (Average)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(1, 0)
    
    plt.tight_layout()
    plt.savefig('mean_face_visualization.png')
    
except ImportError:
    print("\n注意: matplotlib未安装，无法可视化")