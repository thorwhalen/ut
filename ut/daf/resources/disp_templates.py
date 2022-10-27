"""Dataframe display tewmplates"""
__author__ = 'thor'


# Inline html table templates
# Note: Generated using
#       templates = get_multiple_template_dicts(
#           '/D/Dropbox/dev/web/templates/inline table formats/Multiple Inline Tables.html')


inline_html_table_template = {
    'box-table-c': {
        'table': {
            'id': 'box-table-c',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'text-align': 'right',
                'summary': '',
            },
        },
        'tbody': {
            'style': {
                'background': '#e8edff',
                'border-bottom': '1px solid #fff',
                'border-top': '1px solid transparent',
                'color': '#669',
                'text-align': 'right',
                'padding': '6px 8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'background': '#b9c9fe',
                'border-bottom': '1px solid #fff',
                'border-top': '4px solid #aabcfe',
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '4px 6px',
            },
        },
    },
    'background-image': {
        'table': {
            'id': 'background-image',
            'style': {
                'background': "url('table-images/blurry.jpg') 330px 59px no-repeat",
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'background': "url('table-images/back.png')",
                'border-top': '1px solid #fff',
                'color': '#669',
                'padding': '9px 12px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'color': '#339',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '12px',
            },
        },
    },
    'box-table-a': {
        'table': {
            'id': 'box-table-a',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'background': '#e8edff',
                'border-bottom': '1px solid #fff',
                'border-top': '1px solid transparent',
                'color': '#669',
                'padding': '8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'background': '#b9c9fe',
                'border-bottom': '1px solid #fff',
                'border-top': '4px solid #aabcfe',
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '8px',
            },
        },
    },
    'box-table-b': {
        'table': {
            'id': 'box-table-b',
            'style': {
                'border-bottom': '7px solid #9baff1',
                'border-collapse': 'collapse',
                'border-top': '7px solid #9baff1',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'center',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'background': '#e8edff',
                'border-left': '1px solid #aabcfe',
                'border-right': '1px solid #aabcfe',
                'color': '#669',
                'padding': '8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'background': '#e8edff',
                'border-left': '1px solid #9baff1',
                'border-right': '1px solid #9baff1',
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '8px',
            },
        },
    },
    'gradient-style': {
        'table': {
            'id': 'gradient-style',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'background': "#e8edff url('table-images/gradback.png') repeat-x",
                'border-bottom': '1px solid #fff',
                'border-top': '1px solid #fff',
                'color': '#669',
                'padding': '8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'background': "#b9c9fe url('table-images/gradhead.png') repeat-x",
                'border-bottom': '1px solid #fff',
                'border-top': '2px solid #d3ddff',
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '8px',
            },
        },
    },
    'hor-minimalist-a': {
        'table': {
            'id': 'hor-minimalist-a',
            'style': {
                'background': '#fff',
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {'style': {'color': '#669', 'padding': '9px 8px 0px 8px'}},
        'thead': {
            'scope': 'col',
            'style': {
                'border-bottom': '2px solid #6678b1',
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '10px 8px',
            },
        },
    },
    'hor-minimalist-b': {
        'table': {
            'id': 'hor-minimalist-b',
            'style': {
                'background': '#fff',
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-bottom': '1px solid #ccc',
                'color': '#669',
                'padding': '6px 8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'border-bottom': '2px solid #6678b1',
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '10px 8px',
            },
        },
    },
    'hor-zebra': {
        'table': {
            'id': 'hor-zebra',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {'style': {'color': '#669', 'padding': '8px'}},
        'thead': {
            'scope': 'col',
            'style': {
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '10px 8px',
            },
        },
    },
    'newspaper-a': {
        'table': {
            'id': 'newspaper-a',
            'style': {
                'border': '1px solid #69c',
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {'style': {'color': '#669', 'padding': '7px 17px 7px 17px'}},
        'thead': {
            'scope': 'col',
            'style': {
                'border-bottom': '1px dashed #69c',
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '12px 17px 12px 17px',
            },
        },
    },
    'newspaper-b': {
        'table': {
            'id': 'newspaper-b',
            'style': {
                'border': '1px solid #69c',
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-top': '1px dashed #fff',
                'color': '#669',
                'padding': '10px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '15px 10px 10px 10px',
            },
        },
    },
    'newspaper-c': {
        'table': {
            'id': 'newspaper-c',
            'style': {
                'border': '1px solid #6cf',
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-right': '1px dashed #6cf',
                'color': '#669',
                'padding': '10px 20px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'border-bottom': '1px solid #fff',
                'border-left': '1px solid #0865c2',
                'border-right': '1px solid #0865c2',
                'border-top': '1px solid #0865c2',
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '20px',
                'text-transform': 'uppercase',
            },
        },
    },
    'one-column-emphasis': {
        'table': {
            'id': 'one-column-emphasis',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-top': '1px solid #e8edff',
                'color': '#669',
                'padding': '10px 15px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '12px 15px',
            },
        },
    },
    'pattern-style-a': {
        'table': {
            'id': 'pattern-style-a',
            'style': {
                'background': "url('table-images/pattern.png')",
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-bottom': '1px solid #fff',
                'border-top': '1px solid transparent',
                'color': '#669',
                'padding': '8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'border-bottom': '1px solid #fff',
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '8px',
            },
        },
    },
    'pattern-style-b': {
        'table': {
            'id': 'pattern-style-b',
            'style': {
                'background': "url('table-images/patternb.png')",
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-bottom': '1px solid #fff',
                'border-top': '1px solid transparent',
                'color': '#669',
                'padding': '8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'border-bottom': '1px solid #fff',
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '8px',
            },
        },
    },
    'rounded-corner': {
        'table': {
            'id': 'rounded-corner',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'background': '#e8edff',
                'border-top': '1px solid #fff',
                'color': '#669',
                'padding': '8px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'background': "#b9c9fe url('table-images/left.png') left -1px no-repeat",
                'color': '#039',
                'font-size': '13px',
                'font-weight': 'normal',
                'padding': '8px',
            },
        },
    },
    'ver-minimalist': {
        'table': {
            'id': 'ver-minimalist',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-left': '30px solid #fff',
                'border-right': '30px solid #fff',
                'color': '#669',
                'padding': '12px 2px 0px 2px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'border-bottom': '2px solid #6678b1',
                'border-left': '30px solid #fff',
                'border-right': '30px solid #fff',
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '8px 2px',
            },
        },
    },
    'ver-zebra': {
        'table': {
            'id': 'ver-zebra',
            'style': {
                'border-collapse': 'collapse',
                'font-family': '"Lucida Sans Unicode", "Lucida Grande", Sans-Serif',
                'font-size': '12px',
                'margin': '45px',
                'text-align': 'left',
                'width': '480px',
            },
            'summary': '',
        },
        'tbody': {
            'style': {
                'border-left': '1px solid #fff',
                'border-right': '1px solid #fff',
                'color': '#669',
                'padding': '8px 15px',
            }
        },
        'thead': {
            'scope': 'col',
            'style': {
                'background': '#dce4ff',
                'border-bottom': '1px solid #d6dfff',
                'border-left': '1px solid #fff',
                'border-right': '1px solid #fff',
                'color': '#039',
                'font-size': '14px',
                'font-weight': 'normal',
                'padding': '12px 15px',
            },
        },
    },
}
