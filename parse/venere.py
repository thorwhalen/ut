__author__ = 'thorwhalen'

import ut.parse.util as parse_util
import re
import ut.pstr.trans as pstr_trans
import ut.parse.bsoup as parse_bsoup

pois_near_hotel_exp_0 = re.compile('(?<=\. )[\w ]+(?=- 0\.\d km / 0\.\d mi)')
pois_near_hotel_exp = re.compile('(.+)- (\d+[\.\d]*) km / (\d+[\.\d]*) mi')


def get_pois_near_hotel_location(html):
    html = parse_util.x_to_soup(html)
    html = html.find('div', attrs={'id': 'location-distances'}).renderContents()
    t = html.split('<br/>')
    t = ['. ' + x for x in t]
    # print len(t)
    # return [re.search(pois_near_hotel_exp, x) for x in t]
    return [
        x.group(0).strip()
        for x in [re.search(pois_near_hotel_exp_0, x) for x in t]
        if x
    ]


def parse_hotel_info_page(html):
    html = parse_util.x_to_soup(html)
    d = dict()

    # hotel name
    d = parse_bsoup.add_text_to_parse_dict(
        soup=html,
        parse_dict=d,
        key='hotel_name',
        name='h1',
        attrs={'property': 'v:name'},
        text_transform=parse_util.strip_spaces,
    )
    # hotel address
    tag = html.find(name='p', attrs={'id': 'property-address'})
    if tag:
        d['hotel_address'] = pstr_trans.strip(tag.text)
        d = parse_bsoup.add_text_to_parse_dict(
            soup=tag,
            parse_dict=d,
            key='hotel_street_address',
            name='span',
            attrs={'property': 'v:street-address'},
            text_transform=parse_util.strip_spaces,
        )
        d = parse_bsoup.add_text_to_parse_dict(
            soup=tag,
            parse_dict=d,
            key='hotel_locality',
            name='span',
            attrs={'property': 'v:locality'},
            text_transform=parse_util.strip_spaces,
        )

    # average price
    d = parse_bsoup.add_text_to_parse_dict(
        soup=html,
        parse_dict=d,
        key='currency',
        name='span',
        attrs={'id': 'currency-symbol'},
        text_transform=parse_util.strip_spaces,
    )
    avgPriceEl0 = html.find(name='span', attrs={'id': 'avgPriceEl0'})
    avgPriceDecimals = html.find(name='sup', attrs={'id': 'avgPriceDecimals'})
    if avgPriceEl0:
        d['average_price'] = avgPriceEl0.text
        if avgPriceDecimals:
            d['average_price'] = d['average_price'] + avgPriceDecimals.text
        d['average_price'] = float(d['average_price'])

    # facebook likes
    d = parse_bsoup.add_text_to_parse_dict(
        soup=html,
        parse_dict=d,
        key='facebook_likes',
        name='span',
        attrs={'class': 'pluginCountTextDisconnected'},
        text_transform=float,
    )

    # num_of_photos
    tag = html.find(name='div', attrs={'id': 'photo_gallery'})
    if tag:
        d['num_of_photos'] = len(tag.findAll(name='li'))

    # hotel description
    d = parse_bsoup.add_text_to_parse_dict(
        soup=html,
        parse_dict=d,
        key='hotel_description',
        name='div',
        attrs={'id': 'hotel-description-body'},
        text_transform=parse_util.strip_spaces,
    )

    # average_venere_rating
    tag = html.find(name='div', attrs={'id': 'avg_guest_rating'})
    if tag:
        d['average_venere_rating'] = float(
            tag.find(name='b', attrs={'property': 'v:rating'}).text
        )

    # facilities
    tag = html.find(name='div', attrs={'id': 'facilities'})
    if tag:
        facilities = tag.findAll(name='li')
        if facilities:
            d['facilities'] = [parse_util.strip_spaces(x.text) for x in facilities]

    # alternate names
    tag = html.find(name='div', attrs={'id': 'also_known_as'})
    if tag:
        tag = tag.find(name='p')
        if tag:
            t = [parse_util.strip_spaces(x) for x in tag.renderContents().split('<br>')]
            t = [parse_util.strip_tags(x) for x in t]
            d['alternate_names'] = t

    # overview_reviews
    tag = html.find(name='div', attrs={'id': 'reviews-overview-hbar-box'})
    if tag:
        tagg = tag.findAll(
            name='div', attrs={'class': 'reviews-overview-horizzontalbar'}
        )
        if tagg:
            d['overview_reviews'] = dict()
            for t in tagg:
                d['overview_reviews'][t.find(name='p').text] = float(
                    t.find(name='b').text
                )

    # location_distances
    tag = html.find(name='div', attrs={'id': 'location-distances'})
    if tag:
        t = re.sub('^[^<]+<h2>.+</h2>', '', tag.renderContents()).split('<br/>')
        tt = [re.findall(pois_near_hotel_exp, x) for x in t]
        tt = [x[0] for x in tt if x]
        d['poi_and_distances'] = [
            {
                'poi': parse_util.strip_spaces(x[0].replace('"', '')),
                'km': float(x[1]),
                'mi': float(x[2]),
            }
            for x in tt
        ]
    return d
