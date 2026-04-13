import sys
import os
import logging

path = {
    'local': {
        'raw_response_matrix': 'data/response_matrix.csv',
        'cached_response_matrix': 'data/response_matrix.pkl'
    },
    'euler': {
        'raw_response_matrix': '/cluster/project/sachan/pencui/ProjectsData/skillset/data/response_matrix.csv',
        'cached_response_matrix': '/cluster/project/sachan/pencui/ProjectsData/skillset/data/response_matrix.pkl'
    }
}

