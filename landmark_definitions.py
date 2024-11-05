
SMPL_LANDMARK_INDICES = {"HEAD_TOP": 412,
                    "HEAD_LEFT_TEMPLE": 166,
                    "NECK_ADAM_APPLE": 3050,
                    "LEFT_HEEL": 3458,
                    "RIGHT_HEEL": 6858,
                    "LEFT_NIPPLE": 3042,
                    "RIGHT_NIPPLE": 6489,

                    "SHOULDER_TOP": 3068,
                    "INSEAM_POINT": 3149,
                    "BELLY_BUTTON": 3501,
                    "BACK_BELLY_BUTTON": 3022,
                    "CROTCH": 1210,
                    "PUBIC_BONE": 3145,
                    "RIGHT_WRIST": 5559,
                    "LEFT_WRIST": 2241,
                    "RIGHT_BICEP": 4855,
                    "RIGHT_FOREARM": 5197,
                    "LEFT_SHOULDER": 3011,
                    "RIGHT_SHOULDER": 6470,
                    "LOW_LEFT_HIP": 3134,
                    "LEFT_THIGH": 947,
                    "LEFT_CALF": 1103,
                    "LEFT_ANKLE": 3325,
                    "LEFT_ELBOW": 1643,

                    "BUTTHOLE": 3119,

                    # introduce CAESAR landmarks because
                    # i need to measure arms in parts
                    "Cervicale": 829,
                    'Rt. Acromion': 5342,
                    'Rt. Humeral Lateral Epicn': 5090,
                    'Rt. Ulnar Styloid': 5520,
                    }

SMPL_LANDMARK_INDICES["HEELS"] = (SMPL_LANDMARK_INDICES["LEFT_HEEL"], 
                                  SMPL_LANDMARK_INDICES["RIGHT_HEEL"])


SMPLX_LANDMARK_INDICES = {"HEAD_TOP": 8976,
                    "HEAD_LEFT_TEMPLE": 1980,
                    "NECK_ADAM_APPLE": 8940, 
                    "LEFT_HEEL": 8847,
                    "RIGHT_HEEL": 8635,
                    "LEFT_NIPPLE": 3572,
                    "RIGHT_NIPPLE": 8340,

                    "SHOULDER_TOP": 5616,
                    "INSEAM_POINT": 5601,
                    "BELLY_BUTTON": 5939,
                    "BACK_BELLY_BUTTON": 5941,
                    "CROTCH": 3797,
                    "PUBIC_BONE": 5949,
                    "RIGHT_WRIST": 7449,
                    "LEFT_WRIST": 4823,
                    "RIGHT_BICEP": 6788, 
                    "RIGHT_FOREARM": 7266,
                    "LEFT_SHOULDER": 4442,
                    "RIGHT_SHOULDER": 7218, 
                    "LOW_LEFT_HIP": 4112, 
                    "LEFT_THIGH": 3577,
                    "LEFT_CALF": 3732,
                    "LEFT_ANKLE": 5880
                    }

SMPLX_LANDMARK_INDICES["HEELS"] = (SMPLX_LANDMARK_INDICES["LEFT_HEEL"], 
                                  SMPLX_LANDMARK_INDICES["RIGHT_HEEL"])