"""AdWords reporting tools"""
__author__ = 'thorwhalen'
# This pfile contains function to deal with AdWords reporting

import datetime
import pandas as pd
import numpy as np
import pickle

from adspygoogle import AdWordsClient

import ut as ms
import ut.aw.manip
import ut.daf.ch as daf_ch
# import ut.misc.erenev.data_source as venere_data_source
import json
import functools
#from datapath import datapath
import ut.pcoll.order_conserving as order_conserving


def excel_report_file_to_df(excel_file, sheetname=0):
    df = pd.read_excel(excel_file, sheetname=sheetname)
    if df.iloc[0, 1] == 'Unnamed: 1':
        cols = np.array(df.iloc[2])  # columns are in row 2 (3rd row), so remember them
        df = df.iloc[3:-1].copy()  # get the data (from row 3, till the penultimate row (last row contains stats))
        df.columns = cols  # insert the the correct columns
    else:
        df.columns = df.iloc[0]
        df = df.iloc[1:-1]
    df = df.reset_index(drop=True)  # reset the index
    ms.aw.manip.process_aw_column_names(df)  # process the column names
    return df


def get_client(clientCustomerId='7998744469'):
    # test :  7998744469
    # other : 5127918221
    # US 03 : 7214411738
    # AU 01 : 3851930085
    clientCustomerId = get_account_id(clientCustomerId)
    headers = {'email': os.getenv('VEN_ADWORDS_EMAIL'),
               'password': os.getenv('VEN_ADWORDS_EMAIL_PASSWORD'),
               'clientCustomerId': clientCustomerId,
               # 'userAgent': 'MethodicSolutions',
               'userAgent': 'Venere',
               'developerToken': os.getenv('VEN_ADWORDS_TOKEN'),
               'validateOnly': 'n',
               'partialFailure': 'n'
    }
    print("Getting client for clientCustomerId={}".format(clientCustomerId))
    return AdWordsClient(headers=headers)


def get_report_downloader(client='test'):
    """
    gets a ReportDownloader for a given client. Client can be:
        - an actual AdWordsClient
        - an account name or id (uses get_account_id to get an id from a name
        -
    """
    if not isinstance(client, AdWordsClient):
        print(client)
        print(type(client))
        if isinstance(client, str):
            client = get_client(clientCustomerId=client)
    return client.GetReportDownloader(version='v201302')


def download_report(report_downloader, report_query_str, download_format='df', thousands=None, dtype=None):
    """
    downloads a report using report_downloader (a ReportDownloader or client) using the given query string
    Outputs a dataframe (default) or a string (default if download_format is not 'df' TSV format) or
    """
    if isinstance(report_downloader, AdWordsClient):
        report_downloader = report_downloader.GetReportDownloader(version='v201302')
    elif isinstance(report_downloader,str):
        report_downloader = get_report_downloader(client=report_downloader)
    if download_format == 'df':
        google_report = report_downloader.DownloadReportWithAwql(report_query_str, 'TSV')
        return report_to_df(google_report, thousands, dtype)
    else:
        return report_downloader.DownloadReportWithAwql(report_query_str, download_format)


def mk_report_query_str(
        varList='default',
        source='SEARCH_QUERY_PERFORMANCE_REPORT',
        start_date=1,
        end_date=datetime.date.today(),
        where_dict={}
):
    """
    Makes a query string that will be input to DownloadReportWithAwql
    """
    # where_dict POSITIVE filter (oblige these to be present, and specify the default values if not specified)
    # MJM: I am removing 'Status':'=ACTIVE' from KEYWORDS_PERFORMANCE_REPORT, as discussed with TW. He will remove it manually
    # from the scoring code.
    if source=='KEYWORDS_PERFORMANCE_REPORT':
        where_dict_filter = {'CampaignStatus':'=ACTIVE', 'AdGroupStatus':'=ENABLED', 'Status':'=ACTIVE', 'IsNegative':'=False'}
    elif source=='KEYWORDS_PERFORMANCE_REPORT_IGNORE_STATUS':
        where_dict_filter = {'CampaignStatus':'=ACTIVE', 'AdGroupStatus':'=ENABLED', 'IsNegative':'=False'}
    elif source=='SEARCH_QUERY_PERFORMANCE_REPORT':
        where_dict_filter = {'CampaignStatus':'=ACTIVE', 'AdGroupStatus':'=ENABLED'}
    elif source=='AD_PERFORMANCE_REPORT':
        where_dict_filter = {'CampaignStatus':'=ACTIVE', 'AdGroupStatus':'=ENABLED'}
    else:
        where_dict_filter = {}

    #components of query
    if isinstance(varList, str):
        if varList.find(',') == -1: # if you do not find a comma get the string listing the vars, using varList as a group name
            varList = get_var_list_str(group=varList,source=source)
    else:
        if not isinstance(varList,list):
            raise ValueError("varList must be a comma seperated string or a list of strings")
        # if not, assume this is a list string to be taken as is
    # map varList names to disp names (the ones expected by google's query language)
    varList = x_to_disp_name(varList)
    if isinstance(varList, list):
        varList = ', '.join(varList) # make a comma separated string from the varList list
    # map where_dict keys to disp names
    where_dict = {x_to_disp_name(k):v for (k,v) in list(where_dict.items())}
    # filter where_dict
    where_dict = dict(where_dict_filter,**where_dict)
    # remember the where_vars
    where_vars = list(where_dict.keys())
    # make the where_str (that will be inserted in the query_str
    where_str = ' AND '.join([k + v for (k,v) in list(where_dict.items())])
    # make the date range string
    date_range = dateRange(start_date, end_date)
    # making the query
    query_str = 'SELECT ' + varList + ' FROM ' + source + ' WHERE ' + where_str + ' DURING ' + date_range
    return query_str


def report_to_df(report, thousands=None, dtype=None):
    """
    make a dataframe from the report
    """

    import pandas as pd
    import tempfile

    tempf = tempfile.NamedTemporaryFile()
    try:
        tempf.write(report)
        tempf.seek(0)
        df = pd.io.parsers.read_csv(tempf, skiprows=1, skipfooter=1, header=1, delimiter='\t', thousands=thousands, dtype=dtype)
        df = daf_ch.ch_col_names(df, x_to_lu_name(list(df.columns)), list(df.columns))
        return df
    finally:
        tempf.close()



def get_df_concatination_of_several_accounts(account_list,varList=None,number_of_days=300):
    if varList is None:
        varList = 'Query, Impressions, AdGroupName, CampaignName, AdGroupStatus'
    report_query_str = mk_report_query_str(varList=varList, source='SEARCH_QUERY_PERFORMANCE_REPORT',start_date=number_of_days)
    df = None
    for a in account_list:
        print("%s: downloading %s" % (datetime.now(),a))
        report_downloader = get_report_downloader(a)
        print("  %s: concatinating" % datetime.now())
        df = pd.concat([df,download_report(report_downloader=report_downloader, report_query_str=report_query_str, download_format='df')])
    return df


########################################################################################################################
# UTILS

def x_to_date(x):
    """
    util to get a datetime.date() formatted date from a flexibly specified date
    (list or datetime.date at the time of this writing)
    """
    if isinstance(x, list) and len(x) == 3:
        return datetime.date(year=x[0], month=x[1], day=x[2])
    elif isinstance(x, datetime.date):
        return x
    else:
        print("Unknown format")
        #TODO: Throw exception


def dateRange(start_date=1, end_date=datetime.date.today()):
    """
    util to get a date range string as a google query expects it
        dateRange(i) (i is an int) returns the range between i days ago to now
        dateRange(x) (x is a float) returns the range from x_to_date(x) to now
        dateRange(x,y) (x is a float) returns the range from x_to_date(x) to from x_to_date(y)
    """
    end_date = x_to_date(end_date)
    if isinstance(start_date, int):
        start_date = end_date - datetime.timedelta(days=start_date)
    else:
        start_date = x_to_date(start_date)
    return '{},{}'.format(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))


# MJM - this method lets you set up a decorator (@lu_name_df) above any method that returns a dataframe and should have
# the dataframe column names converted to lu names
def lu_name_df(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        df = func(*args, **kwargs)
        df.columns = x_to_lu_name(df.columns) # replace column names with lu names
        return df
    return inner

########################################################################################################################
# PARAMETERS

def x_to_lu_name(var):
    """
    returns a (copy) of the list var where all recognized variable names are converted to their lu_name forms
    """
    return _convert_name(var,lu_name_x_dict)


def x_to_xml_name(var):
    """
    returns a (copy) of the list var where all recognized variable names are converted to their xml name forms
    """
    return _convert_name(var, xml_x_dict)

def x_to_disp_name(var):
    """
    returns a (copy) of the list var where all recognized variable names are converted to their disp name forms
    """
    return _convert_name(var,disp_x_dict)

def _convert_name(list_of_names, mapping_dict):
    """
    returns a (copy) of the list var where all recognized variable names are converted to a standardized name specified
    by the keys of the mapping_dict
    """
    if isinstance(list_of_names,str):
        type_of_input_list = 'string'
        list_of_names = [x.strip() for x in list_of_names.split(',')]
    else:
        if not isinstance(list_of_names, list):
            list(list_of_names)
        type_of_input_list = 'list'
        list_of_names = list(list_of_names) # a copy of var
    # else:
    #     raise TypeError("input must be a comma separated string or a list")
    for vi, v in enumerate(list_of_names):
        for key, value in mapping_dict.items():
            if v in value:
                list_of_names[vi] = key
                continue
    if type_of_input_list == 'list':
        return list_of_names
    else: # assuming type_of_input_list was a string
        return ', '.join(list_of_names)


def get_var_list_str(group='default', source='SEARCH_QUERY_PERFORMANCE_REPORT'):
    if source == 'SEARCH_QUERY_PERFORMANCE_REPORT':
        var_list = {
            'default': ('Query, AdGroupId, AdGroupName, AveragePosition, '
                        'CampaignId, CampaignName, Clicks, Cost, Impressions, KeywordId, KeywordTextMatchingQuery, '
                        'MatchType, MatchTypeWithVariant'),
            'q_kmv_picc': ('Query, AveragePosition, KeywordTextMatchingQuery, MatchType, MatchTypeWithVariant, '
                           'AveragePosition, Impressions, Clicks, Cost'),
            'q_km_picc': 'Query, KeywordTextMatchingQuery, MatchType, AveragePosition, Impressions, Clicks, Cost',
            'q_matts_pick': ('Query, Impressions, AdGroupName, CampaignName, '
                             'KeywordTextMatchingQuery, MatchType, '
                            'Cost, AveragePosition, Clicks, AdGroupId, CampaignId, KeywordId'),
            'q_picc': 'Query, AveragePosition, Impressions, Clicks, Cost',
            'q_ipic': 'Query, KeywordId, Impressions, AveragePosition, Clicks',
            'q_iipic': 'Query, AdGroupId, KeywordId, Impressions, AveragePosition, Clicks'
        }
    elif source == 'AD_PERFORMANCE_REPORT':
        var_list = {
            'default':
                ['AdGroupName', 'AdGroupId', 'Headline', 'Description1', 'Description2', 'CreativeDestinationUrl', 'CampaignId'],
            'ad_elements':
                ['AdGroupName', 'AdGroupId', 'Headline', 'Description1', 'Description2', 'CreativeDestinationUrl', 'CampaignId']
        }
    elif source == 'KEYWORDS_PERFORMANCE_REPORT':
        var_list = {
            'default': ('KeywordText, KeywordMatchType, Max. CPC, AdGroupName, CampaignName, '
                        'Clicks, Cost, Impressions, AveragePosition, Id, AdGroupId, CampaignId'),
            'kw_elements': ['KeywordText', 'KeywordMatchType', 'MaxCpc', 'Id', 'AdGroupId', 'CampaignId',
                            'Clicks', 'Cost', 'Impressions', 'AveragePosition', 'DestinationUrl', 'Status'],
            'q_kw_perf': 'KeywordText, KeywordMatchType, AveragePosition, Impressions, Clicks, Cost, AdGroupId',
            'q_01': ('KeywordText, KeywordMatchType, Max. CPC, AdGroupName, CampaignName, '
                        'Clicks, Cost, Impressions, AveragePosition, Id, AdGroupId, CampaignId'),
            'q_static_attributes': ['KeywordText', 'KeywordMatchType', 'AdGroupName', 'CampaignName',
                                  'Id', 'AdGroupId', 'CampaignId'],
            'q_main_attributes': ['KeywordText', 'KeywordMatchType', 'MaxCpc', 'AdGroupName', 'CampaignName',
                                  'Id', 'AdGroupId', 'CampaignId'],
            'q_kmv_picc': ('KeywordText, KeywordMatchType, Max. CPC, '
                           'Clicks, Cost, Impressions, AveragePosition, Status'),
            'q_kms': 'KeywordText, KeywordMatchType, Status'
        }
    else:
        print("sorry, I'm not aware of the source name!")

    if group == 'groups':
        return var_list
    else:
        return var_list.get(group, ('')) # empty is the group is not found


def print_some_report_sources():
    source_list = [
        'SEARCH_QUERY_PERFORMANCE_REPORT'
        'KEYWORDS_PERFORMANCE_REPORT'
        'AD_PERFORMANCE_REPORT'
        'ADGROUP_PERFORMANCE_REPORT'
        'CAMPAIGN_PERFORMANCE_REPORT'
        'ACCOUNT_PERFORMANCE_REPORT'
        'CLICK_PERFORMANCE_REPORT'
        'DESTINATION_URL_REPORT'
        'GEO_PERFORMANCE_REPORT'
        'URL_PERFORMANCE_REPORT'
    ]
    for source in source_list: print(source)


def import_account_str_to_id(source='/D/Dropbox/dev/py/data/aw/account_name_accountid.csv',
                             target='/D/Dropbox/dev/py/data/aw/account_name_accountid.p'):
    """
    This function imports a csv into a pickled dict that maps account names to account numbers (customer ids)
    """
    df = pd.read_csv('/D/Dropbox/dev/py/data/aw/account_name_accountid.csv')
    df.index = df['Account']
    del df['Account']
    dfdict = df.to_dict()
    pickle.dump(dfdict, open(target, "wb"))


def get_account_id(account='test', account_str_to_id_dict=''):
    """
    Once the account_str_id map is imported using import_account_str_to_id(),
    you can use the get_account_id() function to grab an account id from an account name
In [70]: get_account_id('AU 01')
Out[70]: 3851930085
ac
The test account id is also there
In [80]: get_account_id('test')
Out[80]: 7998744469

In fact, it's the default
In [82]: get_account_id()
Out[82]: 7998744469

If you input an empty string as the account name, the function will print a list of available account names
In [83]: get_account_id('')
AVAILABLE ACCOUNT NAMES:
['FI 01-INDIE', 'ZH 01', 'IT 01-CONT', 'UK 01-INDIE', etc.]

You'll get this list also if you enter an account name that is not available
In [86]: get_account_id('matt is a monkey')
THIS ACCOUNT NAME IS NOT AVAILABLE! AVAILABLE ACCOUNTS:
['FI 01-INDIE', 'ZH 01', 'IT 01-CONT', 'UK 01-INDIE', etc.]

If you ask for the "account" 'dict', the function will output the dict mapping account names (keys) to account ids (values)
In [44]: get_account_id('dict')
Out[44]:
{'AU 01': 3851930085,
 'DE 01': 1194790660,
 'DE 01-ALL': 6985472731, etc.}
    """

    # if not account_str_to_id_dict:
    #     account_str_to_id_dict = venere_data_source.account_str_to_id_dict

    if not isinstance(account_str_to_id_dict, dict):
        raise ValueError("account_str_to_id_dict must be a dict")
        # if isinstance(account_str_to_id_dict, string):
        #     account_str_to_id_dict = pickle.load(open(account_str_to_id_dict, "rb"))
        # else:
        #     print "Unknown account_str_to_id_dict type"
        #     return dict() # empty dict
    if not account:
        print("AVAILABLE ACCOUNT NAMES:")
        print(list(account_str_to_id_dict['Customer ID'].keys()))
        return dict() # empty dict
    elif account == 'dict':
        return account_str_to_id_dict['Customer ID']
    elif account in list(account_str_to_id_dict['Customer ID'].values()):
        return account
    elif account not in account_str_to_id_dict['Customer ID']:
        print("THIS ACCOUNT NAME IS NOT AVAILABLE! AVAILABLE ACCOUNTS:")
        print(list(account_str_to_id_dict['Customer ID'].keys()))
        return dict() # empty dict
    else:
        return account_str_to_id_dict['Customer ID'][account]


lu_name_x_dict = {
    'a_ce_split': ['aCESplit', 'ACE split'],
    'account': ['account', 'Account'],
    'account_id': ['accountID', 'Account ID'],
    'ad': ['ad', 'Ad'],
    'ad_approval_status': ['adApprovalStatus', 'Ad Approval Status'],
    'ad_extension_id': ['adExtensionID', 'Ad Extension ID'],
    'ad_extension_type': ['adExtensionType', 'Ad Extension Type'],
    'ad_group': ['adGroup', 'Ad group', 'advertentiegroep'],
    'ad_group_id': ['adGroupID', 'Ad group ID', 'adGroupId'],
    'ad_group_state': ['adGroupState', 'Ad group state'],
    'ad_id': ['adID', 'Ad ID'],
    'ad_state': ['adState', 'Ad state'],
    'ad_type': ['adType', 'Ad type'],
    'added': ['added', 'Added'],
    'approval_status': ['approvalStatus', 'Approval Status'],
    'attribute_values': ['attributeValues', 'Attribute Values'],
    'audience': ['audience', 'Audience'],
    'audience_state': ['audienceState', 'Audience state'],
    'avg_cpc': ['avgCPC', 'Avg. CPC', 'Gem. CPC'],
    'avg_cpm': ['avgCPM', 'Avg. CPM'],
    'avg_cpp': ['avgCPP', 'Avg. CPP'],
    'avg_position': ['avgPosition', 'Avg. position', 'Gem. positie'],
    'bidding_strategy': ['biddingStrategy', 'Bidding strategy'],
    'budget': ['budget', 'Budget'],
    'budget_explicitly_shared': ['budgetExplicitlyShared',
                                  'Budget explicitly shared'],
    'budget_id': ['budgetID', 'Budget ID'],
    'budget_name': ['budgetName', 'Budget Name'],
    'budget_period': ['budgetPeriod', 'Budget period'],
    'budget_state': ['budgetState', 'Budget state'],
    'budget_usage': ['budgetUsage', 'Budget usage'],
    'business_phone_number': ['businessPhoneNumber', 'Business phone number'],
    'cpc_ace_indicator': ['cPCACEIndicator', 'CPC ACE indicator'],
    'cpm_ace_indicator': ['cPMACEIndicator', 'CPM ACE indicator'],
    'ctr_ace_indicator': ['cTRACEIndicator', 'CTR ACE indicator'],
    'call_fee': ['callFee', 'Call fee'],
    'caller_area_code': ['callerAreaCode', 'Caller area code'],
    'caller_country_code': ['callerCountryCode', 'Caller country code'],
    'campaign': ['campaign', 'Campaign'],
    'campaign_id': ['campaignID', 'campaignId', 'Campaign ID', 'Campaign Id'],
    'campaign_name': ['campaignName', 'Campaign Name'],
    'campaign_state': ['campaignState', 'Campaign state'],
    'campaigns': ['campaigns', '# Campaigns'],
    'categories': ['categories', 'Categories'],
    'city': ['city', 'City'],
    'click_id': ['clickId', 'Click Id'],
    'click_type': ['clickType', 'Click type'],
    'clicks': ['clicks', 'Clicks', 'Aantal klikken'],
    'clicks_ace_indicator': ['clicksACEIndicator', 'Clicks ACE indicator'],
    'client_name': ['clientName', 'Client name'],
    'company_name': ['companyName', 'Company name', 'Campagne'],
    'content_impr_share': ['contentImprShare', 'Content Impr. share'],
    'content_lost_is_budget': ['contentLostISBudget',
                                'Content Lost IS (budget)'],
    'content_lost_is_rank': ['contentLostISRank', 'Content Lost IS (rank)'],
    'conv': ['conv', 'Conv.', 'Conversies'],
    'converted_clicks': ['convertedClicks', 'Converted clicks', 'Geconverteerde klikken'],
    'conv1_per_click': ['conv1PerClick', 'Conv. (1-per-click)'],
    'conv1_per_click_ace_indicator': ['conv1PerClickACEIndicator',
                                       'Conv. (1-per-click) ACE indicator'],
    'conv_many_per_click': ['convManyPerClick', 'Conv. (many-per-click)'],
    'conv_many_per_click_ace_indicator': ['convManyPerClickACEIndicator',
                                           'Conv. (many-per-click) ACE indicator'],
    'conv_rate': ['convRate', 'Conv. rate'],
    'conv_rate1_per_click': ['convRate1PerClick', 'Conv. rate (1-per-click)'],
    'conv_rate1_per_click_ace_indicator': ['convRate1PerClickACEIndicator',
                                            'Conv. rate (1-per-click) ACE indicator'],
    'conv_rate_many_per_click': ['convRateManyPerClick',
                                  'Conv. rate (many-per-click)'],
    'conv_rate_many_per_click_ace_indicator': ['convRateManyPerClickACEIndicator',
                                                'Conv. rate (many-per-click) ACE indicator'],
    'conversion_action_name': ['conversionActionName',
                                'Conversion action name'],
    'conversion_optimizer_bid_type': ['conversionOptimizerBidType',
                                       'Conversion optimizer bid type'],
    'conversion_tracker_id': ['conversionTrackerId', 'Conversion Tracker Id'],
    'conversion_tracking_purpose': ['conversionTrackingPurpose',
                                     'Conversion tracking purpose'],
    'cost': ['cost', 'Cost', 'kosten'],
    'cost_ace_indicator': ['costACEIndicator', 'Cost ACE indicator'],
    'cost_conv1_per_click': ['costConv1PerClick',
                              'Cost / conv. (1-per-click)'],
    'cost_conv1_per_click_ace_indicator': ['costConv1PerClickACEIndicator',
                                            'Cost/conv. (1-per-click) ACE indicator'],
    'cost_conv_many_per_click': ['costConvManyPerClick',
                                  'Cost / conv. (many-per-click)'],
    'cost_conv_many_per_click_ace_indicator': ['costConvManyPerClickACEIndicator',
                                                'Cost/conv. (many-per-click) ACE indicator'],
    'country_territory': ['countryTerritory', 'Country/Territory'],
    'criteria_display_name': ['criteriaDisplayName', 'Criteria Display Name'],
    'criteria_type': ['criteriaType', 'Criteria Type'],
    'criterion_id': ['criterionID', 'criterionId', 'Criterion ID'],
    'ctr': ['ctr', 'CTR'],
    'currency': ['currency', 'Currency'],
    'customer_id': ['customerID', 'Customer ID'],
    'day': ['day', 'Day'],
    'day_of_week': ['dayOfWeek', 'Day of week'],
    'default_max_cpc': ['defaultMaxCPC', 'Default max. CPC'],
    'delivery_method': ['deliveryMethod', 'Delivery method'],
    'description_line1': ['descriptionLine1', 'Description line 1'],
    'description_line2': ['descriptionLine2', 'Description line 2'],
    'destination_url': ['destinationURL', 'destinationUrl', 'Destination URL', 'Destination Url', 'CreativeDestinationUrl'],
    'device': ['device', 'Device'],
    'device_preference': ['devicePreference', 'Device preference'],
    'display_network_max_cpc': ['displayNetworkMaxCPC',
                                 'Display Network max. CPC'],
    'display_url': ['displayURL', 'Display URL'],
    'domain': ['domain', 'Domain'],
    'duration_seconds': ['durationSeconds', 'Duration (seconds)'],
    'dynamic_ad_target': ['dynamicAdTarget', 'Dynamic ad target'],
    'dynamically_generated_headline': ['dynamicallyGeneratedHeadline',
                                        'Dynamically generated Headline'],
    'end_time': ['endTime', 'End time'],
    'enhanced': ['enhanced', 'Enhanced'],
    'enhanced_cpc_enabled': ['enhancedCPCEnabled', 'Enhanced CPC enabled'],
    'excluded': ['excluded', 'Excluded'],
    'exclusion': ['exclusion', 'Exclusion'],
    'explicitly_shared': ['explicitlyShared', 'Explicitly shared'],
    'feed_id': ['feedID', 'Feed ID'],
    'feed_item_id': ['feedItemID', 'Feed item ID'],
    'feed_item_status': ['feedItemStatus', 'Feed item status'],
    'feed_placeholder_type': ['feedPlaceholderType', 'Feed placeholder type'],
    'first_level_sub_categories': ['firstLevelSubCategories',
                                    'First level sub-categories'],
    'first_page_cpc': ['firstPageCPC', 'First page CPC'],
    'free_click_rate': ['freeClickRate', 'Free click rate'],
    'free_click_type': ['freeClickType', 'Free click type'],
    'free_clicks': ['freeClicks', 'Free clicks'],
    'frequency': ['frequency', 'Frequency'],
    'highest_position': ['highestPosition', 'Highest position'],
    'hour_of_day': ['hourOfDay', 'Hour of day'],
    'image_ad_name': ['imageAdName', 'Image ad name'],
    'image_hosting_key': ['imageHostingKey', 'Image hosting key'],
    'impressions': ['impressions', 'Impressions', 'Vertoningen'],
    'impressions_ace_indicator': ['impressionsACEIndicator',
                                   'Impressions ACE indicator'],
    'invalid_click_rate': ['invalidClickRate', 'Invalid click rate'],
    'invalid_clicks': ['invalidClicks', 'Invalid clicks'],
    'is_negative': ['isNegative', 'Is negative'],
    'is_targetable': ['isTargetable', 'Is Targetable'],
    'keyword': ['keyword', 'Keyword', 'Zoekterm'],
    'keyword_id': ['keywordID', 'Keyword ID'],
    'keyword_max_cpc': ['keywordMaxCPC', 'Keyword max CPC'],
    'keyword_placement': ['keywordPlacement', 'Keyword / Placement'],
    'keyword_placement_destination_url': ['keywordPlacementDestinationURL',
                                           'Keyword/Placement destination URL'],
    'keyword_placement_state': ['keywordPlacementState',
                                 'Keyword/Placement state'],
    'keyword_state': ['keywordState', 'Keyword state'],
    'keyword_text': ['keywordText', 'Keyword text'],
    'landing_page_title': ['landingPageTitle', 'Landing Page Title'],
    'location': ['location', 'Location'],
    'location_extension_source': ['locationExtensionSource',
                                   'Location Extension Source'],
    'location_type': ['locationType', 'Location type'],
    'login_email': ['loginEmail', 'Login email'],
    'lowest_position': ['lowestPosition', 'Lowest position'],
    'match_type': ['matchType', 'Match type', 'zoektype'],
    'max_cpa': ['maxCPA', 'Max. CPA%'],
    'max_cpa1_per_click': ['maxCPA1PerClick', 'Max. CPA (1-per-click)'],
    'max_cpc': ['maxCPC', 'Max. CPC', 'maxCpc'],
    'max_cpc_source': ['maxCPCSource', 'Max CPC source'],
    'max_cpm': ['maxCPM', 'Max. CPM'],
    'max_cpm_source': ['maxCPMSource', 'Max CPM Source'],
    'max_cpp': ['maxCPP', 'Max. CPP'],
    'member_count': ['memberCount', 'Member Count'],
    'metro_area': ['metroArea', 'Metro area'],
    'month': ['month', 'Month'],
    'month_of_year': ['monthOfYear', 'Month of Year'],
    'most_specific_location': ['mostSpecificLocation',
                                'Most specific location'],
    'negative_keyword': ['negativeKeyword', 'Negative keyword'],
    'network': ['network', 'Network'],
    'network_with_search_partners': ['networkWithSearchPartners',
                                      'Network (with search partners)'],
    'page': ['page', 'Page'],
    'phone_bid_type': ['phoneBidType', 'Phone bid type'],
    'phone_calls': ['phoneCalls', 'Phone calls'],
    'phone_cost': ['phoneCost', 'Phone cost'],
    'phone_impressions': ['phoneImpressions', 'Phone impressions'],
    'placement': ['placement', 'Placement'],
    'placement_state': ['placementState', 'Placement state'],
    'position_ace_indicator': ['positionACEIndicator',
                                'Position ACE indicator'],
    'ptr': ['ptr', 'PTR'],
    'quality_score': ['qualityScore', 'Quality score'],
    'quarter': ['quarter', 'Quarter'],
    'reference_count': ['referenceCount', 'Reference Count'],
    'region': ['region', 'Region'],
    'relative_ctr': ['relativeCTR', 'Relative CTR'],
    'search_exact_match_is': ['searchExactMatchIS', 'Search Exact match IS'],
    'search_impr_share': ['searchImprShare', 'Search Impr. share'],
    'search_lost_is_budget': ['searchLostISBudget', 'Search Lost IS (budget)'],
    'search_lost_is_rank': ['searchLostISRank', 'Search Lost IS (rank)'],
    'search_term': ['searchTerm', 'Search term'],
    'second_level_sub_categories': ['secondLevelSubCategories',
                                     'Second level sub-categories'],
    'shared_set_id': ['sharedSetID', 'Shared Set ID'],
    'shared_set_name': ['sharedSetName', 'Shared Set Name'],
    'shared_set_type': ['sharedSetType', 'Shared Set Type'],
    'start_time': ['startTime', 'Start time'],
    'state': ['state', 'State'],
    'status': ['status', 'Status'],
    'targeting_mode': ['targetingMode', 'Targeting Mode'],
    'this_extension_vs_other': ['thisExtensionVsOther',
                                 'This extension vs. Other'],
    'time_zone': ['timeZone', 'Time zone'],
    'top_level_categories': ['topLevelCategories', 'Top level categories'],
    'top_of_page_cpc': ['topOfPageCPC', 'Top of page CPC'],
    'top_vs_side': ['topVsSide', 'Top vs. side'],
    'topic': ['topic', 'Topic'],
    'topic_state': ['topicState', 'Topic state'],
    'total_conv_value': ['totalConvValue', 'Total conv. value'],
    'total_cost': ['totalCost', 'Total cost'],
    'unique_users': ['uniqueUsers', 'Unique Users'],
    'url': ['url', 'URL'],
    'user_status': ['userStatus'],
    'value_conv1_per_click': ['valueConv1PerClick',
                               'Value / conv. (1-per-click)'],
    'value_conv_many_per_click': ['valueConvManyPerClick',
                                   'Value / conv. (many-per-click)'],
    'view_through_conv': ['viewThroughConv', 'View-through conv.'],
    'view_through_conv_ace_indicator': ['viewThroughConvACEIndicator',
                                         'View-through conv. ACE indicator'],
    'week': ['week', 'Week'],
    'year': ['year', 'Year']}

xml_x_dict = {
    'aCESplit': ['ACE split', 'a_ce_split'],
    'account': ['Account', 'account'],
    'accountID': ['Account ID', 'account_id'],
    'ad': ['Ad', 'ad'],
    'adApprovalStatus': ['Ad Approval Status', 'ad_approval_status'],
    'adExtensionID': ['Ad Extension ID', 'ad_extension_id'],
    'adExtensionType': ['Ad Extension Type', 'ad_extension_type'],
    'adGroup': ['Ad group', 'ad_group'],
    'adGroupID': ['Ad group ID', 'Ad group Id', 'ad_group_id'],
    'adGroupState': ['Ad group state', 'ad_group_state'],
    'adID': ['Ad ID', 'ad_id'],
    'adState': ['Ad state', 'ad_state'],
    'adType': ['Ad type', 'ad_type'],
    'added': ['Added', 'added'],
    'approvalStatus': ['Approval Status', 'approval_status'],
    'attributeValues': ['Attribute Values', 'attribute_values'],
    'audience': ['Audience', 'audience'],
    'audienceState': ['Audience state', 'audience_state'],
    'avgCPC': ['Avg. CPC', 'avg_cpc'],
    'avgCPM': ['Avg. CPM', 'avg_cpm'],
    'avgCPP': ['Avg. CPP', 'avg_cpp'],
    'avgPosition': ['Avg. position', 'avg_position'],
    'biddingStrategy': ['Bidding strategy', 'bidding_strategy'],
    'budget': ['Budget', 'budget'],
    'budgetExplicitlyShared': ['Budget explicitly shared',
                                'budget_explicitly_shared'],
    'budgetID': ['Budget ID', 'budget_id'],
    'budgetName': ['Budget Name', 'budget_name'],
    'budgetPeriod': ['Budget period', 'budget_period'],
    'budgetState': ['Budget state', 'budget_state'],
    'budgetUsage': ['Budget usage', 'budget_usage'],
    'businessPhoneNumber': ['Business phone number', 'business_phone_number'],
    'cPCACEIndicator': ['CPC ACE indicator', 'cpc_ace_indicator'],
    'cPMACEIndicator': ['CPM ACE indicator', 'cpm_ace_indicator'],
    'cTRACEIndicator': ['CTR ACE indicator', 'ctr_ace_indicator'],
    'callFee': ['Call fee', 'call_fee'],
    'callerAreaCode': ['Caller area code', 'caller_area_code'],
    'callerCountryCode': ['Caller country code', 'caller_country_code'],
    'campaign': ['Campaign', 'campaign'],
    'campaignID': ['Campaign ID', 'campaign_id'],
    'campaignState': ['Campaign state', 'campaign_state'],
    'campaigns': ['# Campaigns', 'campaigns'],
    'categories': ['Categories', 'categories'],
    'city': ['City', 'city'],
    'clickId': ['Click Id', 'click_id'],
    'clickType': ['Click type', 'click_type'],
    'clicks': ['Clicks', 'clicks'],
    'clicksACEIndicator': ['Clicks ACE indicator', 'clicks_ace_indicator'],
    'clientName': ['Client name', 'client_name'],
    'companyName': ['Company name', 'company_name'],
    'contentImprShare': ['Content Impr. share', 'content_impr_share'],
    'contentLostISBudget': ['Content Lost IS (budget)',
                             'content_lost_is_budget'],
    'contentLostISRank': ['Content Lost IS (rank)', 'content_lost_is_rank'],
    'conv': ['Conv.', 'conv'],
    'conv1PerClick': ['Conv. (1-per-click)', 'conv1_per_click'],
    'conv1PerClickACEIndicator': ['Conv. (1-per-click) ACE indicator',
                                   'conv1_per_click_ace_indicator'],
    'convManyPerClick': ['Conv. (many-per-click)', 'conv_many_per_click'],
    'convManyPerClickACEIndicator': ['Conv. (many-per-click) ACE indicator',
                                      'conv_many_per_click_ace_indicator'],
    'convRate': ['Conv. rate', 'conv_rate'],
    'convRate1PerClick': ['Conv. rate (1-per-click)', 'conv_rate1_per_click'],
    'convRate1PerClickACEIndicator': ['Conv. rate (1-per-click) ACE indicator',
                                       'conv_rate1_per_click_ace_indicator'],
    'convRateManyPerClick': ['Conv. rate (many-per-click)',
                              'conv_rate_many_per_click'],
    'convRateManyPerClickACEIndicator': ['Conv. rate (many-per-click) ACE indicator',
                                          'conv_rate_many_per_click_ace_indicator'],
    'conversionActionName': ['Conversion action name',
                              'conversion_action_name'],
    'conversionOptimizerBidType': ['Conversion optimizer bid type',
                                    'conversion_optimizer_bid_type'],
    'conversionTrackerId': ['Conversion Tracker Id', 'conversion_tracker_id'],
    'conversionTrackingPurpose': ['Conversion tracking purpose',
                                   'conversion_tracking_purpose'],
    'cost': ['Cost', 'cost'],
    'costACEIndicator': ['Cost ACE indicator', 'cost_ace_indicator'],
    'costConv1PerClick': ['Cost / conv. (1-per-click)',
                           'cost_conv1_per_click'],
    'costConv1PerClickACEIndicator': ['Cost/conv. (1-per-click) ACE indicator',
                                       'cost_conv1_per_click_ace_indicator'],
    'costConvManyPerClick': ['Cost / conv. (many-per-click)',
                              'cost_conv_many_per_click'],
    'costConvManyPerClickACEIndicator': ['Cost/conv. (many-per-click) ACE indicator',
                                          'cost_conv_many_per_click_ace_indicator'],
    'countryTerritory': ['Country/Territory', 'country_territory'],
    'criteriaDisplayName': ['Criteria Display Name', 'criteria_display_name'],
    'criteriaType': ['Criteria Type', 'criteria_type'],
    'criterionID': ['Criterion ID', 'Criterion Id', 'criterion_id'],
    'ctr': ['CTR', 'ctr'],
    'currency': ['Currency', 'currency'],
    'customerID': ['Customer ID', 'customer_id'],
    'day': ['Day', 'day'],
    'dayOfWeek': ['Day of week', 'day_of_week'],
    'defaultMaxCPC': ['Default max. CPC', 'default_max_cpc'],
    'deliveryMethod': ['Delivery method', 'delivery_method'],
    'descriptionLine1': ['Description line 1', 'description_line1'],
    'descriptionLine2': ['Description line 2', 'description_line2'],
    'destinationURL': ['Destination URL', 'destination_url'],
    'device': ['Device', 'device'],
    'devicePreference': ['Device preference', 'device_preference'],
    'displayNetworkMaxCPC': ['Display Network max. CPC',
                              'display_network_max_cpc'],
    'displayURL': ['Display URL', 'display_url'],
    'domain': ['Domain', 'domain'],
    'durationSeconds': ['Duration (seconds)', 'duration_seconds'],
    'dynamicAdTarget': ['Dynamic ad target', 'dynamic_ad_target'],
    'dynamicallyGeneratedHeadline': ['Dynamically generated Headline',
                                      'dynamically_generated_headline'],
    'endTime': ['End time', 'end_time'],
    'enhanced': ['Enhanced', 'enhanced'],
    'enhancedCPCEnabled': ['Enhanced CPC enabled', 'enhanced_cpc_enabled'],
    'excluded': ['Excluded', 'excluded'],
    'exclusion': ['Exclusion', 'exclusion'],
    'explicitlyShared': ['Explicitly shared', 'explicitly_shared'],
    'feedID': ['Feed ID', 'feed_id'],
    'feedItemID': ['Feed item ID', 'feed_item_id'],
    'feedItemStatus': ['Feed item status', 'feed_item_status'],
    'feedPlaceholderType': ['Feed placeholder type', 'feed_placeholder_type'],
    'firstLevelSubCategories': ['First level sub-categories',
                                 'first_level_sub_categories'],
    'firstPageCPC': ['First page CPC', 'first_page_cpc'],
    'freeClickRate': ['Free click rate', 'free_click_rate'],
    'freeClickType': ['Free click type', 'free_click_type'],
    'freeClicks': ['Free clicks', 'free_clicks'],
    'frequency': ['Frequency', 'frequency'],
    'highestPosition': ['Highest position', 'highest_position'],
    'hourOfDay': ['Hour of day', 'hour_of_day'],
    'imageAdName': ['Image ad name', 'image_ad_name'],
    'imageHostingKey': ['Image hosting key', 'image_hosting_key'],
    'impressions': ['Impressions', 'impressions'],
    'impressionsACEIndicator': ['Impressions ACE indicator',
                                 'impressions_ace_indicator'],
    'invalidClickRate': ['Invalid click rate', 'invalid_click_rate'],
    'invalidClicks': ['Invalid clicks', 'invalid_clicks'],
    'isNegative': ['Is negative', 'is_negative'],
    'isTargetable': ['Is Targetable', 'is_targetable'],
    'keyword': ['Keyword', 'keyword'],
    'keywordID': ['Keyword ID', 'keyword_id'],
    'keywordMaxCPC': ['Keyword max CPC', 'keyword_max_cpc'],
    'keywordPlacement': ['Keyword / Placement', 'keyword_placement'],
    'keywordPlacementDestinationURL': ['Keyword/Placement destination URL',
                                        'keyword_placement_destination_url'],
    'keywordPlacementState': ['Keyword/Placement state',
                               'keyword_placement_state'],
    'keywordState': ['Keyword state', 'keyword_state'],
    'keywordText': ['Keyword text', 'keyword_text'],
    'landingPageTitle': ['Landing Page Title', 'landing_page_title'],
    'location': ['Location', 'location'],
    'locationExtensionSource': ['Location Extension Source',
                                 'location_extension_source'],
    'locationType': ['Location type', 'location_type'],
    'loginEmail': ['Login email', 'login_email'],
    'lowestPosition': ['Lowest position', 'lowest_position'],
    'matchType': ['Match type', 'match_type'],
    'maxCPA': ['Max. CPA%', 'max_cpa'],
    'maxCPA1PerClick': ['Max. CPA (1-per-click)', 'max_cpa1_per_click'],
    'maxCPC': ['Max. CPC', 'max_cpc'],
    'maxCPCSource': ['Max CPC source', 'max_cpc_source'],
    'maxCPM': ['Max. CPM', 'max_cpm'],
    'maxCPMSource': ['Max CPM Source', 'max_cpm_source'],
    'maxCPP': ['Max. CPP', 'max_cpp'],
    'memberCount': ['Member Count', 'member_count'],
    'metroArea': ['Metro area', 'metro_area'],
    'month': ['Month', 'month'],
    'monthOfYear': ['Month of Year', 'month_of_year'],
    'mostSpecificLocation': ['Most specific location',
                              'most_specific_location'],
    'negativeKeyword': ['Negative keyword', 'negative_keyword'],
    'network': ['Network', 'network'],
    'networkWithSearchPartners': ['Network (with search partners)',
                                   'network_with_search_partners'],
    'page': ['Page', 'page'],
    'phoneBidType': ['Phone bid type', 'phone_bid_type'],
    'phoneCalls': ['Phone calls', 'phone_calls'],
    'phoneCost': ['Phone cost', 'phone_cost'],
    'phoneImpressions': ['Phone impressions', 'phone_impressions'],
    'placement': ['Placement', 'placement'],
    'placementState': ['Placement state', 'placement_state'],
    'positionACEIndicator': ['Position ACE indicator',
                              'position_ace_indicator'],
    'ptr': ['PTR', 'ptr'],
    'qualityScore': ['Quality score', 'quality_score'],
    'quarter': ['Quarter', 'quarter'],
    'referenceCount': ['Reference Count', 'reference_count'],
    'region': ['Region', 'region'],
    'relativeCTR': ['Relative CTR', 'relative_ctr'],
    'searchExactMatchIS': ['Search Exact match IS', 'search_exact_match_is'],
    'searchImprShare': ['Search Impr. share', 'search_impr_share'],
    'searchLostISBudget': ['Search Lost IS (budget)', 'search_lost_is_budget'],
    'searchLostISRank': ['Search Lost IS (rank)', 'search_lost_is_rank'],
    'searchTerm': ['Search term', 'search_term'],
    'secondLevelSubCategories': ['Second level sub-categories',
                                  'second_level_sub_categories'],
    'sharedSetID': ['Shared Set ID', 'shared_set_id'],
    'sharedSetName': ['Shared Set Name', 'shared_set_name'],
    'sharedSetType': ['Shared Set Type', 'shared_set_type'],
    'startTime': ['Start time', 'start_time'],
    'state': ['State', 'state'],
    'status': ['Status', 'status'],
    'targetingMode': ['Targeting Mode', 'targeting_mode'],
    'thisExtensionVsOther': ['This extension vs. Other',
                              'this_extension_vs_other'],
    'timeZone': ['Time zone', 'time_zone'],
    'topLevelCategories': ['Top level categories', 'top_level_categories'],
    'topOfPageCPC': ['Top of page CPC', 'top_of_page_cpc'],
    'topVsSide': ['Top vs. side', 'top_vs_side'],
    'topic': ['Topic', 'topic'],
    'topicState': ['Topic state', 'topic_state'],
    'totalConvValue': ['Total conv. value', 'total_conv_value'],
    'totalCost': ['Total cost', 'total_cost'],
    'uniqueUsers': ['Unique Users', 'unique_users'],
    'url': ['URL', 'url'],
    'valueConv1PerClick': ['Value / conv. (1-per-click)',
                            'value_conv1_per_click'],
    'valueConvManyPerClick': ['Value / conv. (many-per-click)',
                               'value_conv_many_per_click'],
    'viewThroughConv': ['View-through conv.', 'view_through_conv'],
    'viewThroughConvACEIndicator': ['View-through conv. ACE indicator',
                                     'view_through_conv_ace_indicator'],
    'week': ['Week', 'week'],
    'year': ['Year', 'year']}

disp_x_dict = {
    '# Campaigns': ['campaigns', 'campaigns'],
    'ACE split': ['a_ce_split', 'aCESplit'],
    'Account': ['account', 'account'],
    'Account ID': ['account_id', 'accountID'],
    'Ad': ['ad', 'ad'],
    'Ad Approval Status': ['ad_approval_status', 'adApprovalStatus'],
    'Ad Extension ID': ['ad_extension_id', 'adExtensionID'],
    'Ad Extension Type': ['ad_extension_type', 'adExtensionType'],
    'Ad ID': ['ad_id', 'adID'],
    'Ad group': ['ad_group', 'adGroup'],
    'Ad group ID': ['ad_group_id', 'adGroupID', 'adGroupId'],
    'Ad group state': ['ad_group_state', 'adGroupState'],
    'Ad state': ['ad_state', 'adState'],
    'Ad type': ['ad_type', 'adType'],
    'Added': ['added', 'added'],
    'Approval Status': ['approval_status', 'approvalStatus'],
    'Attribute Values': ['attribute_values', 'attributeValues'],
    'Audience': ['audience', 'audience'],
    'Audience state': ['audience_state', 'audienceState'],
    'Avg. CPC': ['avg_cpc', 'avgCPC'],
    'Avg. CPM': ['avg_cpm', 'avgCPM'],
    'Avg. CPP': ['avg_cpp', 'avgCPP'],
    'Avg. position': ['avg_position', 'avgPosition'],
    'Bidding strategy': ['bidding_strategy', 'biddingStrategy'],
    'Budget': ['budget', 'budget'],
    'Budget ID': ['budget_id', 'budgetID'],
    'Budget Name': ['budget_name', 'budgetName'],
    'Budget explicitly shared': ['budget_explicitly_shared',
                                  'budgetExplicitlyShared'],
    'Budget period': ['budget_period', 'budgetPeriod'],
    'Budget state': ['budget_state', 'budgetState'],
    'Budget usage': ['budget_usage', 'budgetUsage'],
    'Business phone number': ['business_phone_number', 'businessPhoneNumber'],
    'CPC ACE indicator': ['cpc_ace_indicator', 'cPCACEIndicator'],
    'CPM ACE indicator': ['cpm_ace_indicator', 'cPMACEIndicator'],
    'CTR': ['ctr', 'ctr'],
    'CTR ACE indicator': ['ctr_ace_indicator', 'cTRACEIndicator'],
    'Call fee': ['call_fee', 'callFee'],
    'Caller area code': ['caller_area_code', 'callerAreaCode'],
    'Caller country code': ['caller_country_code', 'callerCountryCode'],
    'Campaign': ['campaign', 'campaign'],
    'Campaign ID': ['campaign_id', 'campaignID', 'campaignId'],
    'Campaign Name': ['campaign_name', 'campaignName'],
    'Campaign state': ['campaign_state', 'campaignState'],
    'Categories': ['categories', 'categories'],
    'City': ['city', 'city'],
    'Click Id': ['click_id', 'clickId'],
    'Click type': ['click_type', 'clickType'],
    'Clicks': ['clicks', 'clicks'],
    'Clicks ACE indicator': ['clicks_ace_indicator', 'clicksACEIndicator'],
    'Client name': ['client_name', 'clientName'],
    'Company name': ['company_name', 'companyName'],
    'Content Impr. share': ['content_impr_share', 'contentImprShare'],
    'Content Lost IS (budget)': ['content_lost_is_budget',
                                  'contentLostISBudget'],
    'Content Lost IS (rank)': ['content_lost_is_rank', 'contentLostISRank'],
    'Conv.': ['conv', 'conv'],
    'Conv. (1-per-click)': ['conv1_per_click', 'conv1PerClick'],
    'Conv. (1-per-click) ACE indicator': ['conv1_per_click_ace_indicator',
                                           'conv1PerClickACEIndicator'],
    'Conv. (many-per-click)': ['conv_many_per_click', 'convManyPerClick'],
    'Conv. (many-per-click) ACE indicator': ['conv_many_per_click_ace_indicator',
                                              'convManyPerClickACEIndicator'],
    'Conv. rate': ['conv_rate', 'convRate'],
    'Conv. rate (1-per-click)': ['conv_rate1_per_click', 'convRate1PerClick'],
    'Conv. rate (1-per-click) ACE indicator': ['conv_rate1_per_click_ace_indicator',
                                                'convRate1PerClickACEIndicator'],
    'Conv. rate (many-per-click)': ['conv_rate_many_per_click',
                                     'convRateManyPerClick'],
    'Conv. rate (many-per-click) ACE indicator': ['conv_rate_many_per_click_ace_indicator',
                                                   'convRateManyPerClickACEIndicator'],
    'Conversion Tracker Id': ['conversion_tracker_id', 'conversionTrackerId'],
    'Conversion action name': ['conversion_action_name',
                                'conversionActionName'],
    'Conversion optimizer bid type': ['conversion_optimizer_bid_type',
                                       'conversionOptimizerBidType'],
    'Conversion tracking purpose': ['conversion_tracking_purpose',
                                     'conversionTrackingPurpose'],
    'Cost': ['cost', 'cost'],
    'Cost / conv. (1-per-click)': ['cost_conv1_per_click',
                                    'costConv1PerClick'],
    'Cost / conv. (many-per-click)': ['cost_conv_many_per_click',
                                       'costConvManyPerClick'],
    'Cost ACE indicator': ['cost_ace_indicator', 'costACEIndicator'],
    'Cost/conv. (1-per-click) ACE indicator': ['cost_conv1_per_click_ace_indicator',
                                                'costConv1PerClickACEIndicator'],
    'Cost/conv. (many-per-click) ACE indicator': ['cost_conv_many_per_click_ace_indicator',
                                                   'costConvManyPerClickACEIndicator'],
    'Country/Territory': ['country_territory', 'countryTerritory'],
    'Criteria Display Name': ['criteria_display_name', 'criteriaDisplayName'],
    'Criteria Type': ['criteria_type', 'criteriaType'],
    'Criterion ID': ['criterion_id', 'criterionID', 'Criterion Id'],
    'Currency': ['currency', 'currency'],
    'Customer ID': ['customer_id', 'customerID'],
    'Day': ['day', 'day'],
    'Day of week': ['day_of_week', 'dayOfWeek'],
    'Default max. CPC': ['default_max_cpc', 'defaultMaxCPC'],
    'Delivery method': ['delivery_method', 'deliveryMethod'],
    'Description line 1': ['description_line1', 'descriptionLine1'],
    'Description line 2': ['description_line2', 'descriptionLine2'],
    'Destination URL': ['destination_url', 'destinationURL'],
    'Device': ['device', 'device'],
    'Device preference': ['device_preference', 'devicePreference'],
    'Display Network max. CPC': ['display_network_max_cpc',
                                  'displayNetworkMaxCPC'],
    'Display URL': ['display_url', 'displayURL'],
    'Domain': ['domain', 'domain'],
    'Duration (seconds)': ['duration_seconds', 'durationSeconds'],
    'Dynamic ad target': ['dynamic_ad_target', 'dynamicAdTarget'],
    'Dynamically generated Headline': ['dynamically_generated_headline',
                                        'dynamicallyGeneratedHeadline'],
    'End time': ['end_time', 'endTime'],
    'Enhanced': ['enhanced', 'enhanced'],
    'Enhanced CPC enabled': ['enhanced_cpc_enabled', 'enhancedCPCEnabled'],
    'Excluded': ['excluded', 'excluded'],
    'Exclusion': ['exclusion', 'exclusion'],
    'Explicitly shared': ['explicitly_shared', 'explicitlyShared'],
    'Feed ID': ['feed_id', 'feedID'],
    'Feed item ID': ['feed_item_id', 'feedItemID'],
    'Feed item status': ['feed_item_status', 'feedItemStatus'],
    'Feed placeholder type': ['feed_placeholder_type', 'feedPlaceholderType'],
    'First level sub-categories': ['first_level_sub_categories',
                                    'firstLevelSubCategories'],
    'First page CPC': ['first_page_cpc', 'firstPageCPC'],
    'Free click rate': ['free_click_rate', 'freeClickRate'],
    'Free click type': ['free_click_type', 'freeClickType'],
    'Free clicks': ['free_clicks', 'freeClicks'],
    'Frequency': ['frequency', 'frequency'],
    'Highest position': ['highest_position', 'highestPosition'],
    'Hour of day': ['hour_of_day', 'hourOfDay'],
    'Image ad name': ['image_ad_name', 'imageAdName'],
    'Image hosting key': ['image_hosting_key', 'imageHostingKey'],
    'Impressions': ['impressions', 'impressions'],
    'Impressions ACE indicator': ['impressions_ace_indicator',
                                   'impressionsACEIndicator'],
    'Invalid click rate': ['invalid_click_rate', 'invalidClickRate'],
    'Invalid clicks': ['invalid_clicks', 'invalidClicks'],
    'Is Targetable': ['is_targetable', 'isTargetable'],
    'Is negative': ['is_negative', 'isNegative'],
    'Keyword': ['keyword', 'keyword'],
    'Keyword / Placement': ['keyword_placement', 'keywordPlacement'],
    'Keyword ID': ['keyword_id', 'keywordID'],
    'Keyword max CPC': ['keyword_max_cpc', 'keywordMaxCPC'],
    'Keyword state': ['keyword_state', 'keywordState'],
    'Keyword text': ['keyword_text', 'keywordText'],
    'Keyword/Placement destination URL': ['keyword_placement_destination_url',
                                           'keywordPlacementDestinationURL'],
    'Keyword/Placement state': ['keyword_placement_state',
                                 'keywordPlacementState'],
    'Landing Page Title': ['landing_page_title', 'landingPageTitle'],
    'Location': ['location', 'location'],
    'Location Extension Source': ['location_extension_source',
                                   'locationExtensionSource'],
    'Location type': ['location_type', 'locationType'],
    'Login email': ['login_email', 'loginEmail'],
    'Lowest position': ['lowest_position', 'lowestPosition'],
    'Match type': ['match_type', 'matchType'],
    'Max CPC source': ['max_cpc_source', 'maxCPCSource'],
    'Max CPM Source': ['max_cpm_source', 'maxCPMSource'],
    'Max. CPA (1-per-click)': ['max_cpa1_per_click', 'maxCPA1PerClick'],
    'Max. CPA%': ['max_cpa', 'maxCPA'],
    'Max. CPC': ['max_cpc', 'maxCPC'],
    'Max. CPM': ['max_cpm', 'maxCPM'],
    'Max. CPP': ['max_cpp', 'maxCPP'],
    'Member Count': ['member_count', 'memberCount'],
    'Metro area': ['metro_area', 'metroArea'],
    'Month': ['month', 'month'],
    'Month of Year': ['month_of_year', 'monthOfYear'],
    'Most specific location': ['most_specific_location',
                                'mostSpecificLocation'],
    'Negative keyword': ['negative_keyword', 'negativeKeyword'],
    'Network': ['network', 'network'],
    'Network (with search partners)': ['network_with_search_partners',
                                        'networkWithSearchPartners'],
    'PTR': ['ptr', 'ptr'],
    'Page': ['page', 'page'],
    'Phone bid type': ['phone_bid_type', 'phoneBidType'],
    'Phone calls': ['phone_calls', 'phoneCalls'],
    'Phone cost': ['phone_cost', 'phoneCost'],
    'Phone impressions': ['phone_impressions', 'phoneImpressions'],
    'Placement': ['placement', 'placement'],
    'Placement state': ['placement_state', 'placementState'],
    'Position ACE indicator': ['position_ace_indicator',
                                'positionACEIndicator'],
    'Quality score': ['quality_score', 'qualityScore'],
    'Quarter': ['quarter', 'quarter'],
    'Reference Count': ['reference_count', 'referenceCount'],
    'Region': ['region', 'region'],
    'Relative CTR': ['relative_ctr', 'relativeCTR'],
    'Search Exact match IS': ['search_exact_match_is', 'searchExactMatchIS'],
    'Search Impr. share': ['search_impr_share', 'searchImprShare'],
    'Search Lost IS (budget)': ['search_lost_is_budget', 'searchLostISBudget'],
    'Search Lost IS (rank)': ['search_lost_is_rank', 'searchLostISRank'],
    'Search term': ['search_term', 'searchTerm'],
    'Second level sub-categories': ['second_level_sub_categories',
                                     'secondLevelSubCategories'],
    'Shared Set ID': ['shared_set_id', 'sharedSetID'],
    'Shared Set Name': ['shared_set_name', 'sharedSetName'],
    'Shared Set Type': ['shared_set_type', 'sharedSetType'],
    'Start time': ['start_time', 'startTime'],
    'State': ['state', 'state'],
    'Status': ['status', 'status'],
    'Targeting Mode': ['targeting_mode', 'targetingMode'],
    'This extension vs. Other': ['this_extension_vs_other',
                                  'thisExtensionVsOther'],
    'Time zone': ['time_zone', 'timeZone'],
    'Top level categories': ['top_level_categories', 'topLevelCategories'],
    'Top of page CPC': ['top_of_page_cpc', 'topOfPageCPC'],
    'Top vs. side': ['top_vs_side', 'topVsSide'],
    'Topic': ['topic', 'topic'],
    'Topic state': ['topic_state', 'topicState'],
    'Total conv. value': ['total_conv_value', 'totalConvValue'],
    'Total cost': ['total_cost', 'totalCost'],
    'URL': ['url', 'url'],
    'Unique Users': ['unique_users', 'uniqueUsers'],
    'Value / conv. (1-per-click)': ['value_conv1_per_click',
                                     'valueConv1PerClick'],
    'Value / conv. (many-per-click)': ['value_conv_many_per_click',
                                        'valueConvManyPerClick'],
    'View-through conv.': ['view_through_conv', 'viewThroughConv'],
    'View-through conv. ACE indicator': ['view_through_conv_ace_indicator',
                                          'viewThroughConvACEIndicator'],
    'Week': ['week', 'week'],
    'Year': ['year', 'year']}


########################################################################################################################
# print if ran
#print "you just ran pak/aw/reporting.py"

########################################################################################################################
# testing

# print mk_report_query_str(varList='q_km_picc',start_date=21)


# get_account_id('dict')
# {'AU 01': 3851930085,
#  'DE 01': 1194790660,
#  'DE 01-ALL': 6985472731,
#  'DE 01-INDIE': 5556267758,
#  'DE 01-ROW': 2444288938,
#  'DE 02': 2556715694,
#  'DE 04': 8972217420,
#  'DK 01': 7505070892,
#  'DK 01-INDIE': 4054652176,
#  'DK 02-INDIE': 1846160625,
#  'DK 03-INDIE': 9520958074,
#  'EN 01-ROW': 9930643268,
#  'EN 01-ROW-INDIE': 4232517800,
#  'EN 02-ROW': 1522899549,
#  'EN 10-ROW': 5584281294,
#  'EN 11-ROW': 7057635394,
#  'ES 01': 9908456190,
#  'ES 01-ALL': 8994980430,
#  'ES 01-CONT': 7874475692,
#  'ES 01-INDIE': 7180305048,
#  'ES 01-ROW': 8198397935,
#  'ES 01-ROW-INDIE': 6340692275,
#  'ES 02': 6005714737,
#  'ES 03': 7197651089,
#  'FI 01': 1296579139,
#  'FI 01-INDIE': 5715846021,
#  'FI 02-INDIE': 9571621125,
#  'FI 03-INDIE': 8861581621,
#  'FI 04-INDIE': 2081740278,
#  'FR 01': 5911041630,
#  'FR 01-ALL': 2269774098,
#  'FR 01-INDIE': 5217295598,
#  'FR 02': 4687005389,
#  'FR 02-INDIE': 4925203694,
#  'FR 03': 7466089112,
#  'FR 03-INDIE': 9467453333,
#  'FR 04-INDIE': 5965796572,
#  'GR 01': 7885244288,
#  'IT 01': 2519329330,
#  'IT 01-ALL': 1681185186,
#  'IT 01-CONT': 6177392492,
#  'IT 01-INDIE': 3274557238,
#  'IT 02': 6885141520,
#  'IT 03': 1322961450,
#  'IT 03-INDIE': 8473689494,
#  'IT 04-INDIE': 6181015380,
#  'IT 08': 2054247047,
#  'JP 01': 1672753368,
#  'NL 01': 9274485992,
#  'NL 01-INDIE': 6859081627,
#  'NO 01': 9313644618,
#  'NO 01-INDIE': 5127918221,
#  'NO 02-INDIE': 9376769080,
#  'NO 03-INDIE': 2030487180,
#  'PL 01': 8995156233,
#  'PT 01': 7897882635,
#  'RU 01': 4088933886,
#  'SE 01': 5293372478,
#  'SE 01-INDIE': 6231052325,
#  'SE 02-INDIE': 4074349225,
#  'SE 03-INDIE': 2664341927,
#  'UK 01': 9543142488,
#  'UK 01-INDIE': 8975012908,
#  'UK 03': 5615378768,
#  'US 01-INDIE': 7938359658,
#  'US 03': 7214411738,
#  'US 05': 5158555927,
#  'ZH 01': 5792026168,
#  'ZH 01-INDIE': 5135270078,
#  'test': 7998744469}


if __name__=="__main__":
    import os
    os.environ['MS_DATA'] = '/D/Dropbox/dev/py/data/'
    report_query_str = 'SELECT Query, KeywordId, Impressions, AveragePosition, Clicks FROM SEARCH_QUERY_PERFORMANCE_REPORT DURING 20130607,20130609'
    report_downloader = get_report_downloader('AU 01')
    df = download_report(report_downloader=report_downloader,report_query_str=report_query_str,download_format='df')
    print(len(df))
