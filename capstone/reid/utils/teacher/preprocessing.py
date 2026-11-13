"""Crop geometry is configurable independently of full-frame resizing."""


def resize_shape(height, width, spec, crop=False):
    short = spec['crop_size'] if crop else spec['short_side']
    maximum = spec['crop_size'] if crop else spec['max_side']
    geometry = spec.get('crop_geometry', 'aspect') if crop else 'aspect'
    if geometry == 'square':
        return short, short
    if geometry != 'aspect':
        raise ValueError(f'Unknown crop geometry {geometry}')
    scale = min(short/min(height, width), maximum/max(height, width))
    return round(height*scale), round(width*scale)
