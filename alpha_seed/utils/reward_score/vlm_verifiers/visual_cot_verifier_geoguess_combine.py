import json
import math
import os
import random
import re
import time

import haversine

from alpha_seed.utils.reward_score.extra_reward import match_visual_cot_format, VLM_ARC_CLIENT
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import BaseVerifier, VerifyResult, ExtractAnswerFailed
from alpha_seed.utils.reward_score.vlm_verifiers.base_verifier import VerifierFailed

COUNTRY_GROUPS = [
    # A
    ["Afghanistan", "islamic republic of afghanistan", "afg"],
    ["Albania", "republic of albania", "shqipëria", "shqiperia"],
    ["Algeria", "people's democratic republic of algeria", "al-jazā'ir", "al-jazair"],
    ["Andorra", "principality of andorra"],
    ["Angola", "republic of angola"],
    ["Antigua and Barbuda", "antigua", "barbuda"],
    ["Argentina", "argentine republic", "república argentina", "republica argentina"],
    ["Armenia", "republic of armenia", "hayastan"],
    ["Australia", "commonwealth of australia", "aus", "straya", "aussie"],
    ["Austria", "republic of austria", "österreich", "osterreich"],
    ["Azerbaijan", "republic of azerbaijan", "azərbaycan", "azerbaycan"],

    # B
    ["Bahamas (the)", "the bahamas", "bahamas", "commonwealth of the bahamas"],
    ["Bahrain", "kingdom of bahrain", "al-baḥrayn", "al-bahrayn"],
    ["Bangladesh", "people's republic of bangladesh"],
    ["Barbados", "bajan"],
    ["Belarus", "republic of belarus", "byelorussia", "belorussia"],
    ["Belgium", "kingdom of belgium", "belgië", "belgique"],
    ["Belize", "british honduras"],
    ["Benin", "republic of benin", "dahomey"],
    ["Bhutan", "kingdom of bhutan", "druk yul"],
    [
        "Bolivia (Plurinational State of)", "bolivia", "plurinational state of bolivia",
        "estado plurinacional de bolivia"
    ],
    ["Bosnia and Herzegovina", "bosnia", "herzegovina", "bih", "bosna i hercegovina"],
    ["Botswana", "republic of botswana"],
    ["Brazil", "federative republic of brazil", "brasil", "república federativa do brasil"],
    ["Brunei Darussalam", "brunei", "nation of brunei", "brunei darussalam"],
    ["Bulgaria", "republic of bulgaria"],
    ["Burkina Faso", "burkina", "upper volta"],
    ["Burundi", "republic of burundi"],

    # C
    ["Cabo Verde", "cape verde", "republic of cabo verde"],
    ["Cambodia", "kingdom of cambodia", "kampuchea"],
    ["Cameroon", "republic of cameroon", "république du cameroun"],
    ["Canada", "dominion of canada", "can", "canuck"],
    ["Central African Republic (the)", "central african republic", "car", "the central african republic"],
    ["Chad", "republic of chad", "tchad"],
    ["Chile", "republic of chile", "república de chile"],
    [
        "China", "people's republic of china", "prc", "mainland china", "zhongguo", "zhōngguó", "Hong Kong",
        "hong kong sar", "hong kong special administrative region", "hksar", "hk", "Macao", "macau", "macao sar",
        "macau special administrative region", "Taiwan (Province of China)", "taiwan", "republic of china",
        "chinese taipei", "formosa"
    ],
    ["Colombia", "republic of colombia", "república de colombia"],
    ["Comoros (the)", "comoros", "union of the comoros", "the comoros"],
    ["Congo (the)", "congo", "republic of the congo", "congo-brazzaville", "the congo"],
    [
        "Congo (the Democratic Republic of the)", "democratic republic of the congo", "drc", "dr congo",
        "congo-kinshasa", "zaire", "the democratic republic of the congo"
    ],
    ["Costa Rica", "republic of costa rica", "república de costa rica"],
    ["Côte d'Ivoire", "cote d'ivoire", "ivory coast", "republic of côte d'ivoire"],
    ["Croatia", "republic of croatia", "hrvatska"],
    ["Cuba", "republic of cuba", "república de cuba"],
    ["Cyprus", "republic of cyprus", "kypros", "kibris"],
    ["Czechia", "czech republic", "česko", "česká republika", "cesko"],

    # D
    [
        "Denmark", "kingdom of denmark", "danmark", "Faroe Islands (the)", "faroe islands", "the faroe islands",
        "føroyar", "foroyar", "Greenland", "kalaallit nunaat"
    ],
    ["Djibouti", "republic of djibouti"],
    ["Dominica", "commonwealth of dominica"],
    ["Dominican Republic (the)", "dominican republic", "the dominican republic", "república dominicana"],

    # E
    ["Ecuador", "republic of ecuador", "república del ecuador"],
    ["Egypt", "arab republic of egypt", "misr", "مصر"],
    ["El Salvador", "republic of el salvador", "república de el salvador"],
    ["Equatorial Guinea", "republic of equatorial guinea"],
    ["Eritrea", "state of eritrea"],
    ["Estonia", "republic of estonia", "eesti"],
    ["Eswatini", "kingdom of eswatini", "swaziland"],
    ["Ethiopia", "federal democratic republic of ethiopia"],

    # F
    ["Fiji", "republic of fiji"],
    ["Finland", "republic of finland", "suomi", "Åland Islands", "aland islands", "åland", "aland"],
    [
        "France", "french republic", "république française", "republique francaise", "French Polynesia",
        "polynésie française", "polynesie francaise", "New Caledonia", "nouvelle-calédonie", "nouvelle-caledonie",
        "French Guiana", "guyane", "guyane française", "guyane francaise", "Réunion", "reunion", "Réunion (France)",
        "île de la réunion", "ile de la reunion", "Martinique", "martinica", "Guadeloupe", "guadalupe",
        "Saint Martin (French part)", "st martin", "saint-martin", "Saint Barthélemy", "st barthelemy",
        "saint-barthélemy", "saint-barthelemy", "Saint Pierre and Miquelon", "st pierre and miquelon",
        "saint-pierre et miquelon", "Wallis and Futuna", "wallis-et-futuna", "wallis et futuna",
        "territory of the wallis and futuna islands", "French Southern and Antarctic Lands",
        "french southern territories", "terres australes et antarctiques françaises"
    ],

    # G
    ["Gabon", "gabonese republic", "république gabonaise"],
    ["Gambia (the)", "gambia", "the gambia", "republic of the gambia"],
    ["Georgia", "საქართველო", "sakartvelo"],
    ["Germany", "federal republic of germany", "deutschland", "bundesrepublik deutschland"],
    ["Ghana", "republic of ghana"],
    ["Greece", "hellenic republic", "elláda", "ellada", "hellas"],
    ["Grenada", "spice isle"],
    ["Guatemala", "republic of guatemala", "república de guatemala"],
    ["Guinea", "republic of guinea", "guinée"],
    ["Guinea-Bissau", "republic of guinea-bissau"],
    ["Guyana", "co-operative republic of guyana"],

    # H
    ["Haiti", "republic of haiti", "république d'haïti", "république d'haiti"],
    ["Holy See (the)", "the holy see", "holy see", "vatican", "vatican city", "vatican city state"],
    ["Honduras", "republic of honduras", "república de honduras"],
    ["Hungary", "magyarország", "magyarorszag"],

    # I
    ["Iceland", "republic of iceland", "ísland", "island"],
    ["India", "republic of india", "bharat", "hindustan"],
    ["Indonesia", "republic of indonesia"],
    ["Iran (Islamic Republic of)", "iran", "islamic republic of iran", "persia"],
    ["Iraq", "republic of iraq"],
    ["Ireland", "republic of ireland", "éire", "eire"],
    ["Israel", "state of israel", "yisra'el", "yisrael"],
    ["Italy", "italian republic", "italia", "repubblica italiana"],

    # J
    ["Jamaica", "jam"],
    ["Japan", "nippon", "nihon", "日本"],
    ["Jordan", "hashemite kingdom of jordan", "al-urdun"],

    # K
    ["Kazakhstan", "republic of kazakhstan", "qazaqstan"],
    ["Kenya", "republic of kenya"],
    ["Kiribati", "republic of kiribati"],
    ["Korea (the Democratic People's Republic of)", "north korea", "democratic people's republic of korea", "dprk"],
    ["Korea (the Republic of)", "south korea", "republic of korea", "korea", "rok", "hanguk"],
    ["Kuwait", "state of kuwait"],
    ["Kyrgyzstan", "kyrgyz republic", "kirghizia"],

    # L
    ["Lao People's Democratic Republic (the)", "laos", "lao", "the lao people's democratic republic"],
    ["Latvia", "republic of latvia", "latvija"],
    ["Lebanon", "lebanese republic", "lubnan"],
    ["Lesotho", "kingdom of lesotho"],
    ["Liberia", "republic of liberia"],
    ["Libya", "state of libya", "libyan arab jamahiriya"],
    ["Liechtenstein", "principality of liechtenstein"],
    ["Lithuania", "republic of lithuania", "lietuva"],
    ["Luxembourg", "grand duchy of luxembourg", "letzebuerg"],

    # M
    ["Madagascar", "republic of madagascar", "malagasy republic"],
    ["Malawi", "republic of malawi", "nyasaland"],
    ["Malaysia", "mys"],
    ["Maldives", "republic of maldives", "dhivehi raajje"],
    ["Mali", "republic of mali"],
    ["Malta", "republic of malta"],
    ["Marshall Islands (the)", "marshall islands", "republic of the marshall islands", "the marshall islands"],
    ["Mauritania", "islamic republic of mauritania"],
    ["Mauritius", "republic of mauritius"],
    ["Mexico", "united mexican states", "méxico", "mexico", "estados unidos mexicanos"],
    ["Micronesia (Federated States of)", "micronesia", "federated states of micronesia", "fsm"],
    ["Moldova (the Republic of)", "moldova", "republic of moldova", "the republic of moldova"],
    ["Monaco", "principality of monaco"],
    ["Mongolia", "mongol uls"],
    ["Montenegro", "crna gora"],
    ["Morocco", "kingdom of morocco", "al-maghrib"],
    ["Mozambique", "republic of mozambique", "moçambique"],
    ["Myanmar", "republic of the union of myanmar", "burma"],

    # N
    ["Namibia", "republic of namibia", "southwest africa"],
    ["Nauru", "republic of nauru"],
    ["Nepal", "federal democratic republic of nepal"],
    [
        "Netherlands (the)", "netherlands", "the netherlands", "holland", "nederland", "kingdom of the netherlands",
        "Aruba", "aw", "Curaçao", "curacao", "Sint Maarten (Dutch part)", "sint maarten", "saint martin (dutch part)",
        "Bonaire, Sint Eustatius and Saba", "bes islands", "caribbean netherlands"
    ],
    ["New Zealand", "nz", "aotearoa", "Cook Islands", "kuki airani", "Niue", "Tokelau"],
    ["Nicaragua", "republic of nicaragua", "república de nicaragua"],
    ["Niger (the)", "niger", "republic of the niger", "the niger"],
    ["Nigeria", "federal republic of nigeria"],
    ["North Macedonia", "republic of north macedonia", "macedonia", "fyrom", "former yugoslav republic of macedonia"],
    ["Norway", "kingdom of norway", "norge", "noreg"],

    # O
    ["Oman", "sultanate of oman"],

    # P
    ["Pakistan", "islamic republic of pakistan"],
    ["Palau", "republic of palau", "belau"],
    ["Palestine, State of", "palestine", "state of palestine", "west bank and gaza", "palestinian territories"],
    ["Panama", "republic of panama", "república de panamá"],
    ["Papua New Guinea", "png", "papua", "independent state of papua new guinea"],
    ["Paraguay", "republic of paraguay", "república del paraguay"],
    ["Peru", "republic of peru", "república del perú", "republica del peru"],
    ["Philippines (the)", "philippines", "the philippines", "republic of the philippines", "pilipinas"],
    ["Poland", "republic of poland", "polska", "rzeczpospolita polska"],
    ["Portugal", "portuguese republic", "república portuguesa", "republica portuguesa"],

    # Q
    ["Qatar", "state of qatar"],

    # R
    ["Romania", "românia", "romania"],
    [
        "Russian Federation (the)", "russia", "russian federation", "the russian federation", "rossiya",
        "rossiyskaya federatsiya"
    ],
    ["Rwanda", "republic of rwanda"],

    # S
    ["Saint Kitts and Nevis", "st. kitts and nevis", "st kitts and nevis"],
    ["Saint Lucia", "st. lucia", "st lucia"],
    ["Saint Vincent and the Grenadines", "st. vincent and the grenadines", "st vincent and the grenadines", "svg"],
    ["Samoa", "independent state of samoa", "western samoa"],
    ["San Marino", "republic of san marino", "serenissima repubblica di san marino"],
    ["Sao Tome and Principe", "são tomé and príncipe", "democratic republic of são tomé and príncipe"],
    ["Saudi Arabia", "kingdom of saudi arabia", "ksa", "saudi"],
    ["Senegal", "republic of senegal", "république du sénégal", "republique du senegal"],
    ["Serbia", "republic of serbia", "republika srbija"],
    ["Seychelles", "republic of seychelles"],
    ["Sierra Leone", "republic of sierra leone"],
    ["Singapore", "republic of singapore", "sing", "sg", "lion city"],
    ["Slovakia", "slovak republic", "slovensko"],
    ["Slovenia", "republic of slovenia", "slovenija"],
    ["Solomon Islands", "sol"],
    ["Somalia", "federal republic of somalia"],
    ["South Africa", "republic of south africa", "rsa", "za", "mzansi"],
    ["South Sudan", "republic of south sudan"],
    ["Spain", "kingdom of spain", "españa", "espana"],
    ["Sri Lanka", "democratic socialist republic of sri lanka", "ceylon"],
    ["Sudan (the)", "sudan", "republic of the sudan", "the sudan"],
    ["Suriname", "republic of suriname", "dutch guiana"],
    ["Sweden", "kingdom of sweden", "sverige"],
    ["Switzerland", "swiss confederation", "schweiz", "suisse", "svizzera", "svizra"],
    ["Syrian Arab Republic (the)", "syria", "syrian arab republic", "the syrian arab republic"],

    # T
    ["Tajikistan", "republic of tajikistan", "tojikiston"],
    ["Tanzania, United Republic of", "tanzania", "united republic of tanzania"],
    ["Thailand", "kingdom of thailand", "siam"],
    ["Timor-Leste", "east timor", "democratic republic of timor-leste"],
    ["Togo", "togolese republic", "république togolaise", "republique togolaise"],
    ["Tonga", "kingdom of tonga"],
    ["Trinidad and Tobago", "trinidad", "tobago", "tt"],
    ["Tunisia", "tunisian republic", "tunis"],
    ["Türkiye", "turkiye", "turkey", "republic of türkiye", "republic of turkey"],
    ["Turkmenistan", "republic of turkmenistan"],
    ["Tuvalu", "ellice islands"],

    # U
    ["Uganda", "republic of uganda"],
    ["Ukraine", "ukraїna", "ukraina"],
    ["United Arab Emirates (the)", "united arab emirates", "uae", "emirates", "the united arab emirates"],
    [
        "United Kingdom of Great Britain and Northern Ireland (the)", "united kingdom", "uk", "great britain",
        "britain", "the united kingdom", "england", "northern ireland", "scotland", "wales", "gb", "Bermuda",
        "somers isles", "Cayman Islands", "cayman", "British Virgin Islands", "bvi", "virgin islands",
        "Turks and Caicos Islands", "tci", "Anguilla", "Gibraltar", "Montserrat", "Pitcairn Islands", "pitcairn",
        "pitcairn, henderson, ducie and oeno islands", "Saint Helena, Ascension and Tristan da Cunha", "saint helena",
        "st helena", "ascension", "tristan da cunha", "British Indian Ocean Territory", "biot", "chagos archipelago",
        "Falkland Islands (the) [Malvinas]", "falkland islands", "malvinas", "the falkland islands"
    ],
    [
        "United States of America (the)", "united states", "usa", "united states of america", "us", "the united states",
        "america", "u.s.a.", "u.s.", "Puerto Rico", "pr", "commonwealth of puerto rico", "Guam", "gu", "guåhan",
        "guahan", "American Samoa", "as", "amerika sāmoa", "amerika samoa", "U.S. Virgin Islands", "us virgin islands",
        "virgin islands of the united states", "usvi", "Northern Mariana Islands", "cnmi",
        "commonwealth of the northern mariana islands"
    ],
    ["Uruguay", "oriental republic of uruguay", "república oriental del uruguay"],
    ["Uzbekistan", "republic of uzbekistan", "o'zbekiston", "ozbekiston"],

    # V
    ["Vanuatu", "republic of vanuatu", "new hebrides"],
    ["Venezuela (Bolivarian Republic of)", "venezuela", "bolivarian republic of venezuela"],
    ["Viet Nam", "vietnam", "socialist republic of vietnam", "vn"],

    # Y
    ["Yemen", "republic of yemen", "al-yaman"],

    # Z
    ["Zambia", "republic of zambia", "northern rhodesia"],
    ["Zimbabwe", "republic of zimbabwe", "southern rhodesia", "rhodesia"]
]

CONTINENT_GROUPS = [
    ["Australia", "Australian continent", "Oceania", "Insular Oceania", "Australian Continent"],
]

COUNTRY_ALIASES = {}
for group in COUNTRY_GROUPS:
    for country_name in group:
        COUNTRY_ALIASES[country_name.lower()] = group

CONTINENT_ALIASES = {}
for group in CONTINENT_GROUPS:
    for continent_name in group:
        CONTINENT_ALIASES[continent_name.lower()] = group


def are_same_continent(country1: str, country2: str) -> bool:
    """
    Check if two country names refer to the same canonical country.
    
    Args:
        country1: First country name to compare
        country2: Second country name to compare
        
    Returns:
        True if both names refer to the same canonical country, False otherwise
    """
    if not country1 or not country2:
        return False

    norm1 = country1.strip().lower()
    norm2 = country2.strip().lower()

    if norm1 == norm2:
        return True

    if norm1 in CONTINENT_ALIASES and norm2 in CONTINENT_ALIASES:
        canonical1 = CONTINENT_ALIASES[norm1][0]
        canonical2 = CONTINENT_ALIASES[norm2][0]

        return canonical1 == canonical2

    return False


def are_same_country(country1: str, country2: str) -> bool:
    """
    Check if two country names refer to the same canonical country.
    
    Args:
        country1: First country name to compare
        country2: Second country name to compare
        
    Returns:
        True if both names refer to the same canonical country, False otherwise
    """
    if not country1 or not country2:
        return False

    norm1 = country1.strip().lower()
    norm2 = country2.strip().lower()

    if norm1 == norm2:
        return True

    if norm1 in COUNTRY_ALIASES and norm2 in COUNTRY_ALIASES:
        canonical1 = COUNTRY_ALIASES[norm1][0]
        canonical2 = COUNTRY_ALIASES[norm2][0]

        return canonical1 == canonical2

    return False


# ANSWER_JUDGE_PROMPT = '''你是一个超强的判题专家。给定一道题目的参考答案和一段学生的回答。你的任务是判断该回答是否符合参考答案。若符合，回答是，否则回答否。请注意你的回答要么是是，要么是否，不能含有其他内容。
# <问题>
# {}
# <学生的回答>
# {}
# <参考答案>
# {}
# '''
# # Original

# ANSWER_JUDGE_PROMPT = '''你是一个超强的判题专家。给定一道题目的参考答案和一段学生的回答。你的任务是判断该回答是否在指代上符合参考答案，不需要拼写完全一样。若符合，回答是，否则回答否。请注意你的回答要么是是，要么是否，不能含有其他内容。
# <问题>
# {}
# <学生的回答>
# {}
# <参考答案>
# {}
# '''
# # v3.3.1

# ANSWER_JUDGE_PROMPT = '''你是一个超强的判题专家。给定一道预测城市的地理题目的参考答案和一段学生的回答。你的任务是判断该回答是否符合参考答案，同一个城市的不同拼写也算符合。若符合，回答是，否则回答否。请注意你的回答要么是是，要么是否，不能含有其他内容。
# <问题>
# {}
# <学生的回答>
# {}
# <参考答案>
# {}
# '''
# # v3.3.2

ANSWER_JUDGE_PROMPT = '''你是一个超强的判题专家。给定一道预测城市的地理题目的参考答案和一段学生的回答。你的任务是判断该回答是否符合参考答案，同一个城市的不同名称、拼写、缩写也算符合，但学生的回答必须是一个准确的城市名称，模糊的描述或者不确定的答案都算不符合。若符合，回答是，否则回答否。请注意你的回答要么是是，要么是否，不能含有其他内容。
<问题>
{}
<学生的回答>
{}
<参考答案>
{}
'''


def answer_judge(query, answer, ref_answer, client, endpoint):
    max_retry_num = 5
    tb = ''
    for _ in range(max_retry_num):
        try:
            completion = client.chat.completions.create(
                model=endpoint,
                messages=[
                    {
                        "role": "user",
                        "content": ANSWER_JUDGE_PROMPT.format(query, answer, ref_answer)
                    },
                ],
                temperature=0.1,
            )
            matched_results_raw = completion.choices[0].message.content
            matched_results_raw = matched_results_raw.strip()
            assert matched_results_raw in ["是", "否"]
            if matched_results_raw == "是":
                score = 1.0
            else:
                score = 0.0
            return score
        except Exception:
            sleep_time = random.random() * 10 + 5
            time.sleep(sleep_time)
            import sys
            import traceback
            tb = "".join(traceback.format_exception(*sys.exc_info()))
            continue
    raise VerifierFailed(message=tb)


def geoguess_score(query, response, gt_answer, client, endpoint):

    gt_city, gt_country, gt_continent, gt_lat, gt_lng = gt_answer

    # calculate distance score
    parse_lat_fail = False
    try:
        lat_match = re.search(r"^\s*(?:\*\*)?(?:L|l)at(?:itude)?(?:\*\*)?(?::|=|\s+)?\s*([-+]?\s*\d+\.?\d*)", response,
                              re.MULTILINE)
        lat = float(lat_match.group(1).strip().replace(' ', ''))

    except (AttributeError, IndexError, ValueError) as e:
        parse_lat_fail = True
        lat = None

    parse_lng_fail = False
    try:
        lng_match = re.search(r"^\s*(?:\*\*)?(?:L|l)(?:ng|ong(?:itude)?)(?:\*\*)?(?::|=|\s+)?\s*([-+]?\s*\d+\.?\d*)",
                              response, re.MULTILINE)
        lng = float(lng_match.group(1).strip().replace(' ', ''))

    except (AttributeError, IndexError, ValueError) as e:
        parse_lng_fail = True
        lng = None

    if lat is not None and lng is not None:

        distance_km = haversine.haversine((gt_lat, gt_lng), (lat, lng))

        if distance_km * 1000 <= 25:
            score = 1.0
        else:
            score = 1.0 * math.pow(0.99866017, distance_km)

    else:
        distance_km = None
        score = 0

    # calculate city correctness
    parse_city_fail = False
    try:
        city_match = re.search(r"^\s*(?:\*\*)?(?:C|c)ity(?:\*\*)?:\s*([^,\r\n]+)", response, re.MULTILINE)
        city = city_match.group(1).strip()
    except (AttributeError, IndexError) as e:
        parse_city_fail = True
        city = None

    if city:
        city_correct = answer_judge(query, city, gt_city, client, endpoint)

    else:
        city_correct = False

    # calculate country correctness
    parse_country_fail = False
    try:
        country_match = re.search(r"^\s*(?:\*\*)?(?:C|c)ountry(?:\*\*)?:\s*([^,\r\n]+)", response, re.MULTILINE)
        country = country_match.group(1).strip()
    except (AttributeError, IndexError) as e:
        parse_country_fail = True
        country = None
    country_correct = (are_same_country(gt_country, country)) if country else False

    # calculate continent correctness
    parse_continent_fail = False
    try:
        continent_match = re.search(r"^\s*(?:\*\*)?(?:C|c)ontinent(?:\*\*)?:\s*([^,\r\n]+)", response, re.MULTILINE)
        continent = continent_match.group(1).strip()
    except (AttributeError, IndexError) as e:
        parse_continent_fail = True
        continent = None
    continent_correct = (are_same_continent(gt_continent, continent)) if continent else False

    parse_error = int(parse_city_fail or parse_country_fail or parse_continent_fail or parse_lat_fail or parse_lng_fail)

    return score, int(city_correct), int(country_correct), int(continent_correct), parse_error


class VisualCoTVerifier_Geo_Combine(BaseVerifier):

    def verify(self, response: str, verifier_feature_dict: dict) -> VerifyResult:
        # 检查FC格式错误
        if not match_visual_cot_format(response, verifier_feature=verifier_feature_dict):
            raise ExtractAnswerFailed

        query = verifier_feature_dict['query']
        answer = verifier_feature_dict['answer']

        if isinstance(answer, str):
            answer = json.loads(answer)

        if not os.environ.get("VLM_ARC_KEY", None):
            raise VerifierFailed(message="There is no VLM_ARC_KEY in env!")
        if VLM_ARC_CLIENT is None:
            raise VerifierFailed(message="No VLM_ARC_CLIENT, likely because there is no VLM_ARC_KEY in env!")

        if not os.environ.get("VLM_ARC_ENDPOINT", None):
            raise VerifierFailed(message="There is no VLM_ARC_ENDPOINT in env!")
        endpoint = os.environ.get("VLM_ARC_ENDPOINT", None)

        try:
            final_answer: str = response.split("</think>")[-1].split(
                "<[EOS_never_used_51bce0c785ca2f68081bfa7d91973934]>")[0]
        except Exception:
            return VerifyResult(score=0, extracted_answer=response)

        if not final_answer.strip():
            raise ExtractAnswerFailed

        # score = answer_judge(query, final_answer, answer, VLM_ARC_CLIENT, endpoint)

        score_distance, city_correct, country_correct, continent_correct, parse_error = geoguess_score(
            query, final_answer, answer, VLM_ARC_CLIENT, endpoint)

        weight_city = 1
        weight_country = 1
        weight_continent = 1

        if continent_correct == 0:
            weight_continent = 0.1
        if country_correct == 0:
            weight_country = 0.25
        if city_correct == 0:
            weight_city = 0.75
        # if city_correct == 0:
        #     weight_city = 0.9

        score = score_distance * weight_continent * weight_country * weight_city

        return VerifyResult(score=score, extracted_answer=final_answer)
