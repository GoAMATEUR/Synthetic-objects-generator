import numpy as np

CATEGORIES = [
    {
        'id': 1,
        'name': 'white_cup',
        'supercategory': 'cup',
    },
    {
        'id': 2,
        'name': 'cyan_cup',
        'supercategory': 'cup',
    },
    {
        'id': 3,
        'name': 'cyan_cup',
        'supercategory': 'cup',
    },
    {
        'id': 4,
        'name': 'apple',
        'supercategory': 'fruit',
    },
    {
        'id': 5,
        'name': 'banana',
        'supercategory': 'fruit',
    }
]

CATEGORIES_COLOR_CONFIGS = {
    "cup": [
        {"class": "white_cup", "color": np.array([0.9, 0.9, 0.9, 1.0])},
        {"class": "cyan_cup", "color": np.array([187, 255, 255, 255]) / 255},                      
        {"class": "green_cup", "color": np.array([127, 255, 0, 255]) / 255},
    ],
    "fruit": [
        {"class": "apple", "color": None},
        {"class": "banana", "color": None}
    ]
}