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
    print "Getting client for clientCustomerId={}".format(clientCustomerId)
    return AdWordsClient(headers=headers)


def get_report_downloader(client='test'):
    """
    gets a ReportDownloader for a given client. Client can be:
        - an actual AdWordsClient
        - an account name or id (uses get_account_id to get an id from a name
        -
    """
    if not isinstance(client, AdWordsClient):
        print client
        print type(client)
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
    elif isinstance(report_downloader,basestring):
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
    if isinstance(varList, basestring):
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
    where_dict = {x_to_disp_name(k):v for (k,v) in where_dict.items()}
    # filter where_dict
    where_dict = dict(where_dict_filter,**where_dict)
    # remember the where_vars
    where_vars = where_dict.keys()
    # make the where_str (that will be inserted in the query_str
    where_str = ' AND '.join([k + v for (k,v) in where_dict.items()])
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
        print "%s: downloading %s" % (datetime.now(),a)
        report_downloader = get_report_downloader(a)
        print "  %s: concatinating" % datetime.now()
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
        print "Unknown format"
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
    if isinstance(list_of_names,basestring):
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
        for key, value in mapping_dict.iteritems():
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
        print "sorry, I'm not aware of the source name!"

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
    for source in source_list: print source


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
        print "AVAILABLE ACCOUNT NAMES:"
        print account_str_to_id_dict['Customer ID'].keys()
        return dict() # empty dict
    elif account == 'dict':
        return account_str_to_id_dict['Customer ID']
    elif account in account_str_to_id_dict['Customer ID'].values():
        return account
    elif not account_str_to_id_dict['Customer ID'].has_key(account):
        print "THIS ACCOUNT NAME IS NOT AVAILABLE! AVAILABLE ACCOUNTS:"
        print account_str_to_id_dict['Customer ID'].keys()
        return dict() # empty dict
    else:
        return account_str_to_id_dict['Customer ID'][account]


lu_name_x_dict = {
    u'a_ce_split': [u'aCESplit', u'ACE split'],
    u'account': [u'account', u'Account'],
    u'account_id': [u'accountID', u'Account ID'],
    u'ad': [u'ad', u'Ad'],
    u'ad_approval_status': [u'adApprovalStatus', u'Ad Approval Status'],
    u'ad_extension_id': [u'adExtensionID', u'Ad Extension ID'],
    u'ad_extension_type': [u'adExtensionType', u'Ad Extension Type'],
    u'ad_group': [u'adGroup', u'Ad group', u'advertentiegroep'],
    u'ad_group_id': [u'adGroupID', u'Ad group ID', u'adGroupId'],
    u'ad_group_state': [u'adGroupState', u'Ad group state'],
    u'ad_id': [u'adID', u'Ad ID'],
    u'ad_state': [u'adState', u'Ad state'],
    u'ad_type': [u'adType', u'Ad type'],
    u'added': [u'added', u'Added'],
    u'approval_status': [u'approvalStatus', u'Approval Status'],
    u'attribute_values': [u'attributeValues', u'Attribute Values'],
    u'audience': [u'audience', u'Audience'],
    u'audience_state': [u'audienceState', u'Audience state'],
    u'avg_cpc': [u'avgCPC', u'Avg. CPC', u'Gem. CPC'],
    u'avg_cpm': [u'avgCPM', u'Avg. CPM'],
    u'avg_cpp': [u'avgCPP', u'Avg. CPP'],
    u'avg_position': [u'avgPosition', u'Avg. position', u'Gem. positie'],
    u'bidding_strategy': [u'biddingStrategy', u'Bidding strategy'],
    u'budget': [u'budget', u'Budget'],
    u'budget_explicitly_shared': [u'budgetExplicitlyShared',
                                  u'Budget explicitly shared'],
    u'budget_id': [u'budgetID', u'Budget ID'],
    u'budget_name': [u'budgetName', u'Budget Name'],
    u'budget_period': [u'budgetPeriod', u'Budget period'],
    u'budget_state': [u'budgetState', u'Budget state'],
    u'budget_usage': [u'budgetUsage', u'Budget usage'],
    u'business_phone_number': [u'businessPhoneNumber', u'Business phone number'],
    u'cpc_ace_indicator': [u'cPCACEIndicator', u'CPC ACE indicator'],
    u'cpm_ace_indicator': [u'cPMACEIndicator', u'CPM ACE indicator'],
    u'ctr_ace_indicator': [u'cTRACEIndicator', u'CTR ACE indicator'],
    u'call_fee': [u'callFee', u'Call fee'],
    u'caller_area_code': [u'callerAreaCode', u'Caller area code'],
    u'caller_country_code': [u'callerCountryCode', u'Caller country code'],
    u'campaign': [u'campaign', u'Campaign'],
    u'campaign_id': [u'campaignID', u'campaignId', u'Campaign ID', u'Campaign Id'],
    u'campaign_name': [u'campaignName', u'Campaign Name'],
    u'campaign_state': [u'campaignState', u'Campaign state'],
    u'campaigns': [u'campaigns', u'# Campaigns'],
    u'categories': [u'categories', u'Categories'],
    u'city': [u'city', u'City'],
    u'click_id': [u'clickId', u'Click Id'],
    u'click_type': [u'clickType', u'Click type'],
    u'clicks': [u'clicks', u'Clicks', u'Aantal klikken'],
    u'clicks_ace_indicator': [u'clicksACEIndicator', u'Clicks ACE indicator'],
    u'client_name': [u'clientName', u'Client name'],
    u'company_name': [u'companyName', u'Company name', u'Campagne'],
    u'content_impr_share': [u'contentImprShare', u'Content Impr. share'],
    u'content_lost_is_budget': [u'contentLostISBudget',
                                u'Content Lost IS (budget)'],
    u'content_lost_is_rank': [u'contentLostISRank', u'Content Lost IS (rank)'],
    u'conv': [u'conv', u'Conv.', u'Conversies'],
    u'converted_clicks': [u'convertedClicks', u'Converted clicks', u'Geconverteerde klikken'],
    u'conv1_per_click': [u'conv1PerClick', u'Conv. (1-per-click)'],
    u'conv1_per_click_ace_indicator': [u'conv1PerClickACEIndicator',
                                       u'Conv. (1-per-click) ACE indicator'],
    u'conv_many_per_click': [u'convManyPerClick', u'Conv. (many-per-click)'],
    u'conv_many_per_click_ace_indicator': [u'convManyPerClickACEIndicator',
                                           u'Conv. (many-per-click) ACE indicator'],
    u'conv_rate': [u'convRate', u'Conv. rate'],
    u'conv_rate1_per_click': [u'convRate1PerClick', u'Conv. rate (1-per-click)'],
    u'conv_rate1_per_click_ace_indicator': [u'convRate1PerClickACEIndicator',
                                            u'Conv. rate (1-per-click) ACE indicator'],
    u'conv_rate_many_per_click': [u'convRateManyPerClick',
                                  u'Conv. rate (many-per-click)'],
    u'conv_rate_many_per_click_ace_indicator': [u'convRateManyPerClickACEIndicator',
                                                u'Conv. rate (many-per-click) ACE indicator'],
    u'conversion_action_name': [u'conversionActionName',
                                u'Conversion action name'],
    u'conversion_optimizer_bid_type': [u'conversionOptimizerBidType',
                                       u'Conversion optimizer bid type'],
    u'conversion_tracker_id': [u'conversionTrackerId', u'Conversion Tracker Id'],
    u'conversion_tracking_purpose': [u'conversionTrackingPurpose',
                                     u'Conversion tracking purpose'],
    u'cost': [u'cost', u'Cost', u'kosten'],
    u'cost_ace_indicator': [u'costACEIndicator', u'Cost ACE indicator'],
    u'cost_conv1_per_click': [u'costConv1PerClick',
                              u'Cost / conv. (1-per-click)'],
    u'cost_conv1_per_click_ace_indicator': [u'costConv1PerClickACEIndicator',
                                            u'Cost/conv. (1-per-click) ACE indicator'],
    u'cost_conv_many_per_click': [u'costConvManyPerClick',
                                  u'Cost / conv. (many-per-click)'],
    u'cost_conv_many_per_click_ace_indicator': [u'costConvManyPerClickACEIndicator',
                                                u'Cost/conv. (many-per-click) ACE indicator'],
    u'country_territory': [u'countryTerritory', u'Country/Territory'],
    u'criteria_display_name': [u'criteriaDisplayName', u'Criteria Display Name'],
    u'criteria_type': [u'criteriaType', u'Criteria Type'],
    u'criterion_id': [u'criterionID', u'criterionId', u'Criterion ID'],
    u'ctr': [u'ctr', u'CTR'],
    u'currency': [u'currency', u'Currency'],
    u'customer_id': [u'customerID', u'Customer ID'],
    u'day': [u'day', u'Day'],
    u'day_of_week': [u'dayOfWeek', u'Day of week'],
    u'default_max_cpc': [u'defaultMaxCPC', u'Default max. CPC'],
    u'delivery_method': [u'deliveryMethod', u'Delivery method'],
    u'description_line1': [u'descriptionLine1', u'Description line 1'],
    u'description_line2': [u'descriptionLine2', u'Description line 2'],
    u'destination_url': [u'destinationURL', u'destinationUrl', u'Destination URL', u'Destination Url', u'CreativeDestinationUrl'],
    u'device': [u'device', u'Device'],
    u'device_preference': [u'devicePreference', u'Device preference'],
    u'display_network_max_cpc': [u'displayNetworkMaxCPC',
                                 u'Display Network max. CPC'],
    u'display_url': [u'displayURL', u'Display URL'],
    u'domain': [u'domain', u'Domain'],
    u'duration_seconds': [u'durationSeconds', u'Duration (seconds)'],
    u'dynamic_ad_target': [u'dynamicAdTarget', u'Dynamic ad target'],
    u'dynamically_generated_headline': [u'dynamicallyGeneratedHeadline',
                                        u'Dynamically generated Headline'],
    u'end_time': [u'endTime', u'End time'],
    u'enhanced': [u'enhanced', u'Enhanced'],
    u'enhanced_cpc_enabled': [u'enhancedCPCEnabled', u'Enhanced CPC enabled'],
    u'excluded': [u'excluded', u'Excluded'],
    u'exclusion': [u'exclusion', u'Exclusion'],
    u'explicitly_shared': [u'explicitlyShared', u'Explicitly shared'],
    u'feed_id': [u'feedID', u'Feed ID'],
    u'feed_item_id': [u'feedItemID', u'Feed item ID'],
    u'feed_item_status': [u'feedItemStatus', u'Feed item status'],
    u'feed_placeholder_type': [u'feedPlaceholderType', u'Feed placeholder type'],
    u'first_level_sub_categories': [u'firstLevelSubCategories',
                                    u'First level sub-categories'],
    u'first_page_cpc': [u'firstPageCPC', u'First page CPC'],
    u'free_click_rate': [u'freeClickRate', u'Free click rate'],
    u'free_click_type': [u'freeClickType', u'Free click type'],
    u'free_clicks': [u'freeClicks', u'Free clicks'],
    u'frequency': [u'frequency', u'Frequency'],
    u'highest_position': [u'highestPosition', u'Highest position'],
    u'hour_of_day': [u'hourOfDay', u'Hour of day'],
    u'image_ad_name': [u'imageAdName', u'Image ad name'],
    u'image_hosting_key': [u'imageHostingKey', u'Image hosting key'],
    u'impressions': [u'impressions', u'Impressions', u'Vertoningen'],
    u'impressions_ace_indicator': [u'impressionsACEIndicator',
                                   u'Impressions ACE indicator'],
    u'invalid_click_rate': [u'invalidClickRate', u'Invalid click rate'],
    u'invalid_clicks': [u'invalidClicks', u'Invalid clicks'],
    u'is_negative': [u'isNegative', u'Is negative'],
    u'is_targetable': [u'isTargetable', u'Is Targetable'],
    u'keyword': [u'keyword', u'Keyword', u'Zoekterm'],
    u'keyword_id': [u'keywordID', u'Keyword ID'],
    u'keyword_max_cpc': [u'keywordMaxCPC', u'Keyword max CPC'],
    u'keyword_placement': [u'keywordPlacement', u'Keyword / Placement'],
    u'keyword_placement_destination_url': [u'keywordPlacementDestinationURL',
                                           u'Keyword/Placement destination URL'],
    u'keyword_placement_state': [u'keywordPlacementState',
                                 u'Keyword/Placement state'],
    u'keyword_state': [u'keywordState', u'Keyword state'],
    u'keyword_text': [u'keywordText', u'Keyword text'],
    u'landing_page_title': [u'landingPageTitle', u'Landing Page Title'],
    u'location': [u'location', u'Location'],
    u'location_extension_source': [u'locationExtensionSource',
                                   u'Location Extension Source'],
    u'location_type': [u'locationType', u'Location type'],
    u'login_email': [u'loginEmail', u'Login email'],
    u'lowest_position': [u'lowestPosition', u'Lowest position'],
    u'match_type': [u'matchType', u'Match type', u'zoektype'],
    u'max_cpa': [u'maxCPA', u'Max. CPA%'],
    u'max_cpa1_per_click': [u'maxCPA1PerClick', u'Max. CPA (1-per-click)'],
    u'max_cpc': [u'maxCPC', u'Max. CPC', u'maxCpc'],
    u'max_cpc_source': [u'maxCPCSource', u'Max CPC source'],
    u'max_cpm': [u'maxCPM', u'Max. CPM'],
    u'max_cpm_source': [u'maxCPMSource', u'Max CPM Source'],
    u'max_cpp': [u'maxCPP', u'Max. CPP'],
    u'member_count': [u'memberCount', u'Member Count'],
    u'metro_area': [u'metroArea', u'Metro area'],
    u'month': [u'month', u'Month'],
    u'month_of_year': [u'monthOfYear', u'Month of Year'],
    u'most_specific_location': [u'mostSpecificLocation',
                                u'Most specific location'],
    u'negative_keyword': [u'negativeKeyword', u'Negative keyword'],
    u'network': [u'network', u'Network'],
    u'network_with_search_partners': [u'networkWithSearchPartners',
                                      u'Network (with search partners)'],
    u'page': [u'page', u'Page'],
    u'phone_bid_type': [u'phoneBidType', u'Phone bid type'],
    u'phone_calls': [u'phoneCalls', u'Phone calls'],
    u'phone_cost': [u'phoneCost', u'Phone cost'],
    u'phone_impressions': [u'phoneImpressions', u'Phone impressions'],
    u'placement': [u'placement', u'Placement'],
    u'placement_state': [u'placementState', u'Placement state'],
    u'position_ace_indicator': [u'positionACEIndicator',
                                u'Position ACE indicator'],
    u'ptr': [u'ptr', u'PTR'],
    u'quality_score': [u'qualityScore', u'Quality score'],
    u'quarter': [u'quarter', u'Quarter'],
    u'reference_count': [u'referenceCount', u'Reference Count'],
    u'region': [u'region', u'Region'],
    u'relative_ctr': [u'relativeCTR', u'Relative CTR'],
    u'search_exact_match_is': [u'searchExactMatchIS', u'Search Exact match IS'],
    u'search_impr_share': [u'searchImprShare', u'Search Impr. share'],
    u'search_lost_is_budget': [u'searchLostISBudget', u'Search Lost IS (budget)'],
    u'search_lost_is_rank': [u'searchLostISRank', u'Search Lost IS (rank)'],
    u'search_term': [u'searchTerm', u'Search term'],
    u'second_level_sub_categories': [u'secondLevelSubCategories',
                                     u'Second level sub-categories'],
    u'shared_set_id': [u'sharedSetID', u'Shared Set ID'],
    u'shared_set_name': [u'sharedSetName', u'Shared Set Name'],
    u'shared_set_type': [u'sharedSetType', u'Shared Set Type'],
    u'start_time': [u'startTime', u'Start time'],
    u'state': [u'state', u'State'],
    u'status': [u'status', u'Status'],
    u'targeting_mode': [u'targetingMode', u'Targeting Mode'],
    u'this_extension_vs_other': [u'thisExtensionVsOther',
                                 u'This extension vs. Other'],
    u'time_zone': [u'timeZone', u'Time zone'],
    u'top_level_categories': [u'topLevelCategories', u'Top level categories'],
    u'top_of_page_cpc': [u'topOfPageCPC', u'Top of page CPC'],
    u'top_vs_side': [u'topVsSide', u'Top vs. side'],
    u'topic': [u'topic', u'Topic'],
    u'topic_state': [u'topicState', u'Topic state'],
    u'total_conv_value': [u'totalConvValue', u'Total conv. value'],
    u'total_cost': [u'totalCost', u'Total cost'],
    u'unique_users': [u'uniqueUsers', u'Unique Users'],
    u'url': [u'url', u'URL'],
    u'user_status': [u'userStatus'],
    u'value_conv1_per_click': [u'valueConv1PerClick',
                               u'Value / conv. (1-per-click)'],
    u'value_conv_many_per_click': [u'valueConvManyPerClick',
                                   u'Value / conv. (many-per-click)'],
    u'view_through_conv': [u'viewThroughConv', u'View-through conv.'],
    u'view_through_conv_ace_indicator': [u'viewThroughConvACEIndicator',
                                         u'View-through conv. ACE indicator'],
    u'week': [u'week', u'Week'],
    u'year': [u'year', u'Year']}

xml_x_dict = {
    u'aCESplit': [u'ACE split', u'a_ce_split'],
    u'account': [u'Account', u'account'],
    u'accountID': [u'Account ID', u'account_id'],
    u'ad': [u'Ad', u'ad'],
    u'adApprovalStatus': [u'Ad Approval Status', u'ad_approval_status'],
    u'adExtensionID': [u'Ad Extension ID', u'ad_extension_id'],
    u'adExtensionType': [u'Ad Extension Type', u'ad_extension_type'],
    u'adGroup': [u'Ad group', u'ad_group'],
    u'adGroupID': [u'Ad group ID', u'Ad group Id', u'ad_group_id'],
    u'adGroupState': [u'Ad group state', u'ad_group_state'],
    u'adID': [u'Ad ID', u'ad_id'],
    u'adState': [u'Ad state', u'ad_state'],
    u'adType': [u'Ad type', u'ad_type'],
    u'added': [u'Added', u'added'],
    u'approvalStatus': [u'Approval Status', u'approval_status'],
    u'attributeValues': [u'Attribute Values', u'attribute_values'],
    u'audience': [u'Audience', u'audience'],
    u'audienceState': [u'Audience state', u'audience_state'],
    u'avgCPC': [u'Avg. CPC', u'avg_cpc'],
    u'avgCPM': [u'Avg. CPM', u'avg_cpm'],
    u'avgCPP': [u'Avg. CPP', u'avg_cpp'],
    u'avgPosition': [u'Avg. position', u'avg_position'],
    u'biddingStrategy': [u'Bidding strategy', u'bidding_strategy'],
    u'budget': [u'Budget', u'budget'],
    u'budgetExplicitlyShared': [u'Budget explicitly shared',
                                u'budget_explicitly_shared'],
    u'budgetID': [u'Budget ID', u'budget_id'],
    u'budgetName': [u'Budget Name', u'budget_name'],
    u'budgetPeriod': [u'Budget period', u'budget_period'],
    u'budgetState': [u'Budget state', u'budget_state'],
    u'budgetUsage': [u'Budget usage', u'budget_usage'],
    u'businessPhoneNumber': [u'Business phone number', u'business_phone_number'],
    u'cPCACEIndicator': [u'CPC ACE indicator', u'cpc_ace_indicator'],
    u'cPMACEIndicator': [u'CPM ACE indicator', u'cpm_ace_indicator'],
    u'cTRACEIndicator': [u'CTR ACE indicator', u'ctr_ace_indicator'],
    u'callFee': [u'Call fee', u'call_fee'],
    u'callerAreaCode': [u'Caller area code', u'caller_area_code'],
    u'callerCountryCode': [u'Caller country code', u'caller_country_code'],
    u'campaign': [u'Campaign', u'campaign'],
    u'campaignID': [u'Campaign ID', u'campaign_id'],
    u'campaignState': [u'Campaign state', u'campaign_state'],
    u'campaigns': [u'# Campaigns', u'campaigns'],
    u'categories': [u'Categories', u'categories'],
    u'city': [u'City', u'city'],
    u'clickId': [u'Click Id', u'click_id'],
    u'clickType': [u'Click type', u'click_type'],
    u'clicks': [u'Clicks', u'clicks'],
    u'clicksACEIndicator': [u'Clicks ACE indicator', u'clicks_ace_indicator'],
    u'clientName': [u'Client name', u'client_name'],
    u'companyName': [u'Company name', u'company_name'],
    u'contentImprShare': [u'Content Impr. share', u'content_impr_share'],
    u'contentLostISBudget': [u'Content Lost IS (budget)',
                             u'content_lost_is_budget'],
    u'contentLostISRank': [u'Content Lost IS (rank)', u'content_lost_is_rank'],
    u'conv': [u'Conv.', u'conv'],
    u'conv1PerClick': [u'Conv. (1-per-click)', u'conv1_per_click'],
    u'conv1PerClickACEIndicator': [u'Conv. (1-per-click) ACE indicator',
                                   u'conv1_per_click_ace_indicator'],
    u'convManyPerClick': [u'Conv. (many-per-click)', u'conv_many_per_click'],
    u'convManyPerClickACEIndicator': [u'Conv. (many-per-click) ACE indicator',
                                      u'conv_many_per_click_ace_indicator'],
    u'convRate': [u'Conv. rate', u'conv_rate'],
    u'convRate1PerClick': [u'Conv. rate (1-per-click)', u'conv_rate1_per_click'],
    u'convRate1PerClickACEIndicator': [u'Conv. rate (1-per-click) ACE indicator',
                                       u'conv_rate1_per_click_ace_indicator'],
    u'convRateManyPerClick': [u'Conv. rate (many-per-click)',
                              u'conv_rate_many_per_click'],
    u'convRateManyPerClickACEIndicator': [u'Conv. rate (many-per-click) ACE indicator',
                                          u'conv_rate_many_per_click_ace_indicator'],
    u'conversionActionName': [u'Conversion action name',
                              u'conversion_action_name'],
    u'conversionOptimizerBidType': [u'Conversion optimizer bid type',
                                    u'conversion_optimizer_bid_type'],
    u'conversionTrackerId': [u'Conversion Tracker Id', u'conversion_tracker_id'],
    u'conversionTrackingPurpose': [u'Conversion tracking purpose',
                                   u'conversion_tracking_purpose'],
    u'cost': [u'Cost', u'cost'],
    u'costACEIndicator': [u'Cost ACE indicator', u'cost_ace_indicator'],
    u'costConv1PerClick': [u'Cost / conv. (1-per-click)',
                           u'cost_conv1_per_click'],
    u'costConv1PerClickACEIndicator': [u'Cost/conv. (1-per-click) ACE indicator',
                                       u'cost_conv1_per_click_ace_indicator'],
    u'costConvManyPerClick': [u'Cost / conv. (many-per-click)',
                              u'cost_conv_many_per_click'],
    u'costConvManyPerClickACEIndicator': [u'Cost/conv. (many-per-click) ACE indicator',
                                          u'cost_conv_many_per_click_ace_indicator'],
    u'countryTerritory': [u'Country/Territory', u'country_territory'],
    u'criteriaDisplayName': [u'Criteria Display Name', u'criteria_display_name'],
    u'criteriaType': [u'Criteria Type', u'criteria_type'],
    u'criterionID': [u'Criterion ID', u'Criterion Id', u'criterion_id'],
    u'ctr': [u'CTR', u'ctr'],
    u'currency': [u'Currency', u'currency'],
    u'customerID': [u'Customer ID', u'customer_id'],
    u'day': [u'Day', u'day'],
    u'dayOfWeek': [u'Day of week', u'day_of_week'],
    u'defaultMaxCPC': [u'Default max. CPC', u'default_max_cpc'],
    u'deliveryMethod': [u'Delivery method', u'delivery_method'],
    u'descriptionLine1': [u'Description line 1', u'description_line1'],
    u'descriptionLine2': [u'Description line 2', u'description_line2'],
    u'destinationURL': [u'Destination URL', u'destination_url'],
    u'device': [u'Device', u'device'],
    u'devicePreference': [u'Device preference', u'device_preference'],
    u'displayNetworkMaxCPC': [u'Display Network max. CPC',
                              u'display_network_max_cpc'],
    u'displayURL': [u'Display URL', u'display_url'],
    u'domain': [u'Domain', u'domain'],
    u'durationSeconds': [u'Duration (seconds)', u'duration_seconds'],
    u'dynamicAdTarget': [u'Dynamic ad target', u'dynamic_ad_target'],
    u'dynamicallyGeneratedHeadline': [u'Dynamically generated Headline',
                                      u'dynamically_generated_headline'],
    u'endTime': [u'End time', u'end_time'],
    u'enhanced': [u'Enhanced', u'enhanced'],
    u'enhancedCPCEnabled': [u'Enhanced CPC enabled', u'enhanced_cpc_enabled'],
    u'excluded': [u'Excluded', u'excluded'],
    u'exclusion': [u'Exclusion', u'exclusion'],
    u'explicitlyShared': [u'Explicitly shared', u'explicitly_shared'],
    u'feedID': [u'Feed ID', u'feed_id'],
    u'feedItemID': [u'Feed item ID', u'feed_item_id'],
    u'feedItemStatus': [u'Feed item status', u'feed_item_status'],
    u'feedPlaceholderType': [u'Feed placeholder type', u'feed_placeholder_type'],
    u'firstLevelSubCategories': [u'First level sub-categories',
                                 u'first_level_sub_categories'],
    u'firstPageCPC': [u'First page CPC', u'first_page_cpc'],
    u'freeClickRate': [u'Free click rate', u'free_click_rate'],
    u'freeClickType': [u'Free click type', u'free_click_type'],
    u'freeClicks': [u'Free clicks', u'free_clicks'],
    u'frequency': [u'Frequency', u'frequency'],
    u'highestPosition': [u'Highest position', u'highest_position'],
    u'hourOfDay': [u'Hour of day', u'hour_of_day'],
    u'imageAdName': [u'Image ad name', u'image_ad_name'],
    u'imageHostingKey': [u'Image hosting key', u'image_hosting_key'],
    u'impressions': [u'Impressions', u'impressions'],
    u'impressionsACEIndicator': [u'Impressions ACE indicator',
                                 u'impressions_ace_indicator'],
    u'invalidClickRate': [u'Invalid click rate', u'invalid_click_rate'],
    u'invalidClicks': [u'Invalid clicks', u'invalid_clicks'],
    u'isNegative': [u'Is negative', u'is_negative'],
    u'isTargetable': [u'Is Targetable', u'is_targetable'],
    u'keyword': [u'Keyword', u'keyword'],
    u'keywordID': [u'Keyword ID', u'keyword_id'],
    u'keywordMaxCPC': [u'Keyword max CPC', u'keyword_max_cpc'],
    u'keywordPlacement': [u'Keyword / Placement', u'keyword_placement'],
    u'keywordPlacementDestinationURL': [u'Keyword/Placement destination URL',
                                        u'keyword_placement_destination_url'],
    u'keywordPlacementState': [u'Keyword/Placement state',
                               u'keyword_placement_state'],
    u'keywordState': [u'Keyword state', u'keyword_state'],
    u'keywordText': [u'Keyword text', u'keyword_text'],
    u'landingPageTitle': [u'Landing Page Title', u'landing_page_title'],
    u'location': [u'Location', u'location'],
    u'locationExtensionSource': [u'Location Extension Source',
                                 u'location_extension_source'],
    u'locationType': [u'Location type', u'location_type'],
    u'loginEmail': [u'Login email', u'login_email'],
    u'lowestPosition': [u'Lowest position', u'lowest_position'],
    u'matchType': [u'Match type', u'match_type'],
    u'maxCPA': [u'Max. CPA%', u'max_cpa'],
    u'maxCPA1PerClick': [u'Max. CPA (1-per-click)', u'max_cpa1_per_click'],
    u'maxCPC': [u'Max. CPC', u'max_cpc'],
    u'maxCPCSource': [u'Max CPC source', u'max_cpc_source'],
    u'maxCPM': [u'Max. CPM', u'max_cpm'],
    u'maxCPMSource': [u'Max CPM Source', u'max_cpm_source'],
    u'maxCPP': [u'Max. CPP', u'max_cpp'],
    u'memberCount': [u'Member Count', u'member_count'],
    u'metroArea': [u'Metro area', u'metro_area'],
    u'month': [u'Month', u'month'],
    u'monthOfYear': [u'Month of Year', u'month_of_year'],
    u'mostSpecificLocation': [u'Most specific location',
                              u'most_specific_location'],
    u'negativeKeyword': [u'Negative keyword', u'negative_keyword'],
    u'network': [u'Network', u'network'],
    u'networkWithSearchPartners': [u'Network (with search partners)',
                                   u'network_with_search_partners'],
    u'page': [u'Page', u'page'],
    u'phoneBidType': [u'Phone bid type', u'phone_bid_type'],
    u'phoneCalls': [u'Phone calls', u'phone_calls'],
    u'phoneCost': [u'Phone cost', u'phone_cost'],
    u'phoneImpressions': [u'Phone impressions', u'phone_impressions'],
    u'placement': [u'Placement', u'placement'],
    u'placementState': [u'Placement state', u'placement_state'],
    u'positionACEIndicator': [u'Position ACE indicator',
                              u'position_ace_indicator'],
    u'ptr': [u'PTR', u'ptr'],
    u'qualityScore': [u'Quality score', u'quality_score'],
    u'quarter': [u'Quarter', u'quarter'],
    u'referenceCount': [u'Reference Count', u'reference_count'],
    u'region': [u'Region', u'region'],
    u'relativeCTR': [u'Relative CTR', u'relative_ctr'],
    u'searchExactMatchIS': [u'Search Exact match IS', u'search_exact_match_is'],
    u'searchImprShare': [u'Search Impr. share', u'search_impr_share'],
    u'searchLostISBudget': [u'Search Lost IS (budget)', u'search_lost_is_budget'],
    u'searchLostISRank': [u'Search Lost IS (rank)', u'search_lost_is_rank'],
    u'searchTerm': [u'Search term', u'search_term'],
    u'secondLevelSubCategories': [u'Second level sub-categories',
                                  u'second_level_sub_categories'],
    u'sharedSetID': [u'Shared Set ID', u'shared_set_id'],
    u'sharedSetName': [u'Shared Set Name', u'shared_set_name'],
    u'sharedSetType': [u'Shared Set Type', u'shared_set_type'],
    u'startTime': [u'Start time', u'start_time'],
    u'state': [u'State', u'state'],
    u'status': [u'Status', u'status'],
    u'targetingMode': [u'Targeting Mode', u'targeting_mode'],
    u'thisExtensionVsOther': [u'This extension vs. Other',
                              u'this_extension_vs_other'],
    u'timeZone': [u'Time zone', u'time_zone'],
    u'topLevelCategories': [u'Top level categories', u'top_level_categories'],
    u'topOfPageCPC': [u'Top of page CPC', u'top_of_page_cpc'],
    u'topVsSide': [u'Top vs. side', u'top_vs_side'],
    u'topic': [u'Topic', u'topic'],
    u'topicState': [u'Topic state', u'topic_state'],
    u'totalConvValue': [u'Total conv. value', u'total_conv_value'],
    u'totalCost': [u'Total cost', u'total_cost'],
    u'uniqueUsers': [u'Unique Users', u'unique_users'],
    u'url': [u'URL', u'url'],
    u'valueConv1PerClick': [u'Value / conv. (1-per-click)',
                            u'value_conv1_per_click'],
    u'valueConvManyPerClick': [u'Value / conv. (many-per-click)',
                               u'value_conv_many_per_click'],
    u'viewThroughConv': [u'View-through conv.', u'view_through_conv'],
    u'viewThroughConvACEIndicator': [u'View-through conv. ACE indicator',
                                     u'view_through_conv_ace_indicator'],
    u'week': [u'Week', u'week'],
    u'year': [u'Year', u'year']}

disp_x_dict = {
    u'# Campaigns': [u'campaigns', u'campaigns'],
    u'ACE split': [u'a_ce_split', u'aCESplit'],
    u'Account': [u'account', u'account'],
    u'Account ID': [u'account_id', u'accountID'],
    u'Ad': [u'ad', u'ad'],
    u'Ad Approval Status': [u'ad_approval_status', u'adApprovalStatus'],
    u'Ad Extension ID': [u'ad_extension_id', u'adExtensionID'],
    u'Ad Extension Type': [u'ad_extension_type', u'adExtensionType'],
    u'Ad ID': [u'ad_id', u'adID'],
    u'Ad group': [u'ad_group', u'adGroup'],
    u'Ad group ID': [u'ad_group_id', u'adGroupID', u'adGroupId'],
    u'Ad group state': [u'ad_group_state', u'adGroupState'],
    u'Ad state': [u'ad_state', u'adState'],
    u'Ad type': [u'ad_type', u'adType'],
    u'Added': [u'added', u'added'],
    u'Approval Status': [u'approval_status', u'approvalStatus'],
    u'Attribute Values': [u'attribute_values', u'attributeValues'],
    u'Audience': [u'audience', u'audience'],
    u'Audience state': [u'audience_state', u'audienceState'],
    u'Avg. CPC': [u'avg_cpc', u'avgCPC'],
    u'Avg. CPM': [u'avg_cpm', u'avgCPM'],
    u'Avg. CPP': [u'avg_cpp', u'avgCPP'],
    u'Avg. position': [u'avg_position', u'avgPosition'],
    u'Bidding strategy': [u'bidding_strategy', u'biddingStrategy'],
    u'Budget': [u'budget', u'budget'],
    u'Budget ID': [u'budget_id', u'budgetID'],
    u'Budget Name': [u'budget_name', u'budgetName'],
    u'Budget explicitly shared': [u'budget_explicitly_shared',
                                  u'budgetExplicitlyShared'],
    u'Budget period': [u'budget_period', u'budgetPeriod'],
    u'Budget state': [u'budget_state', u'budgetState'],
    u'Budget usage': [u'budget_usage', u'budgetUsage'],
    u'Business phone number': [u'business_phone_number', u'businessPhoneNumber'],
    u'CPC ACE indicator': [u'cpc_ace_indicator', u'cPCACEIndicator'],
    u'CPM ACE indicator': [u'cpm_ace_indicator', u'cPMACEIndicator'],
    u'CTR': [u'ctr', u'ctr'],
    u'CTR ACE indicator': [u'ctr_ace_indicator', u'cTRACEIndicator'],
    u'Call fee': [u'call_fee', u'callFee'],
    u'Caller area code': [u'caller_area_code', u'callerAreaCode'],
    u'Caller country code': [u'caller_country_code', u'callerCountryCode'],
    u'Campaign': [u'campaign', u'campaign'],
    u'Campaign ID': [u'campaign_id', u'campaignID', u'campaignId'],
    u'Campaign Name': [u'campaign_name', u'campaignName'],
    u'Campaign state': [u'campaign_state', u'campaignState'],
    u'Categories': [u'categories', u'categories'],
    u'City': [u'city', u'city'],
    u'Click Id': [u'click_id', u'clickId'],
    u'Click type': [u'click_type', u'clickType'],
    u'Clicks': [u'clicks', u'clicks'],
    u'Clicks ACE indicator': [u'clicks_ace_indicator', u'clicksACEIndicator'],
    u'Client name': [u'client_name', u'clientName'],
    u'Company name': [u'company_name', u'companyName'],
    u'Content Impr. share': [u'content_impr_share', u'contentImprShare'],
    u'Content Lost IS (budget)': [u'content_lost_is_budget',
                                  u'contentLostISBudget'],
    u'Content Lost IS (rank)': [u'content_lost_is_rank', u'contentLostISRank'],
    u'Conv.': [u'conv', u'conv'],
    u'Conv. (1-per-click)': [u'conv1_per_click', u'conv1PerClick'],
    u'Conv. (1-per-click) ACE indicator': [u'conv1_per_click_ace_indicator',
                                           u'conv1PerClickACEIndicator'],
    u'Conv. (many-per-click)': [u'conv_many_per_click', u'convManyPerClick'],
    u'Conv. (many-per-click) ACE indicator': [u'conv_many_per_click_ace_indicator',
                                              u'convManyPerClickACEIndicator'],
    u'Conv. rate': [u'conv_rate', u'convRate'],
    u'Conv. rate (1-per-click)': [u'conv_rate1_per_click', u'convRate1PerClick'],
    u'Conv. rate (1-per-click) ACE indicator': [u'conv_rate1_per_click_ace_indicator',
                                                u'convRate1PerClickACEIndicator'],
    u'Conv. rate (many-per-click)': [u'conv_rate_many_per_click',
                                     u'convRateManyPerClick'],
    u'Conv. rate (many-per-click) ACE indicator': [u'conv_rate_many_per_click_ace_indicator',
                                                   u'convRateManyPerClickACEIndicator'],
    u'Conversion Tracker Id': [u'conversion_tracker_id', u'conversionTrackerId'],
    u'Conversion action name': [u'conversion_action_name',
                                u'conversionActionName'],
    u'Conversion optimizer bid type': [u'conversion_optimizer_bid_type',
                                       u'conversionOptimizerBidType'],
    u'Conversion tracking purpose': [u'conversion_tracking_purpose',
                                     u'conversionTrackingPurpose'],
    u'Cost': [u'cost', u'cost'],
    u'Cost / conv. (1-per-click)': [u'cost_conv1_per_click',
                                    u'costConv1PerClick'],
    u'Cost / conv. (many-per-click)': [u'cost_conv_many_per_click',
                                       u'costConvManyPerClick'],
    u'Cost ACE indicator': [u'cost_ace_indicator', u'costACEIndicator'],
    u'Cost/conv. (1-per-click) ACE indicator': [u'cost_conv1_per_click_ace_indicator',
                                                u'costConv1PerClickACEIndicator'],
    u'Cost/conv. (many-per-click) ACE indicator': [u'cost_conv_many_per_click_ace_indicator',
                                                   u'costConvManyPerClickACEIndicator'],
    u'Country/Territory': [u'country_territory', u'countryTerritory'],
    u'Criteria Display Name': [u'criteria_display_name', u'criteriaDisplayName'],
    u'Criteria Type': [u'criteria_type', u'criteriaType'],
    u'Criterion ID': [u'criterion_id', u'criterionID', u'Criterion Id'],
    u'Currency': [u'currency', u'currency'],
    u'Customer ID': [u'customer_id', u'customerID'],
    u'Day': [u'day', u'day'],
    u'Day of week': [u'day_of_week', u'dayOfWeek'],
    u'Default max. CPC': [u'default_max_cpc', u'defaultMaxCPC'],
    u'Delivery method': [u'delivery_method', u'deliveryMethod'],
    u'Description line 1': [u'description_line1', u'descriptionLine1'],
    u'Description line 2': [u'description_line2', u'descriptionLine2'],
    u'Destination URL': [u'destination_url', u'destinationURL'],
    u'Device': [u'device', u'device'],
    u'Device preference': [u'device_preference', u'devicePreference'],
    u'Display Network max. CPC': [u'display_network_max_cpc',
                                  u'displayNetworkMaxCPC'],
    u'Display URL': [u'display_url', u'displayURL'],
    u'Domain': [u'domain', u'domain'],
    u'Duration (seconds)': [u'duration_seconds', u'durationSeconds'],
    u'Dynamic ad target': [u'dynamic_ad_target', u'dynamicAdTarget'],
    u'Dynamically generated Headline': [u'dynamically_generated_headline',
                                        u'dynamicallyGeneratedHeadline'],
    u'End time': [u'end_time', u'endTime'],
    u'Enhanced': [u'enhanced', u'enhanced'],
    u'Enhanced CPC enabled': [u'enhanced_cpc_enabled', u'enhancedCPCEnabled'],
    u'Excluded': [u'excluded', u'excluded'],
    u'Exclusion': [u'exclusion', u'exclusion'],
    u'Explicitly shared': [u'explicitly_shared', u'explicitlyShared'],
    u'Feed ID': [u'feed_id', u'feedID'],
    u'Feed item ID': [u'feed_item_id', u'feedItemID'],
    u'Feed item status': [u'feed_item_status', u'feedItemStatus'],
    u'Feed placeholder type': [u'feed_placeholder_type', u'feedPlaceholderType'],
    u'First level sub-categories': [u'first_level_sub_categories',
                                    u'firstLevelSubCategories'],
    u'First page CPC': [u'first_page_cpc', u'firstPageCPC'],
    u'Free click rate': [u'free_click_rate', u'freeClickRate'],
    u'Free click type': [u'free_click_type', u'freeClickType'],
    u'Free clicks': [u'free_clicks', u'freeClicks'],
    u'Frequency': [u'frequency', u'frequency'],
    u'Highest position': [u'highest_position', u'highestPosition'],
    u'Hour of day': [u'hour_of_day', u'hourOfDay'],
    u'Image ad name': [u'image_ad_name', u'imageAdName'],
    u'Image hosting key': [u'image_hosting_key', u'imageHostingKey'],
    u'Impressions': [u'impressions', u'impressions'],
    u'Impressions ACE indicator': [u'impressions_ace_indicator',
                                   u'impressionsACEIndicator'],
    u'Invalid click rate': [u'invalid_click_rate', u'invalidClickRate'],
    u'Invalid clicks': [u'invalid_clicks', u'invalidClicks'],
    u'Is Targetable': [u'is_targetable', u'isTargetable'],
    u'Is negative': [u'is_negative', u'isNegative'],
    u'Keyword': [u'keyword', u'keyword'],
    u'Keyword / Placement': [u'keyword_placement', u'keywordPlacement'],
    u'Keyword ID': [u'keyword_id', u'keywordID'],
    u'Keyword max CPC': [u'keyword_max_cpc', u'keywordMaxCPC'],
    u'Keyword state': [u'keyword_state', u'keywordState'],
    u'Keyword text': [u'keyword_text', u'keywordText'],
    u'Keyword/Placement destination URL': [u'keyword_placement_destination_url',
                                           u'keywordPlacementDestinationURL'],
    u'Keyword/Placement state': [u'keyword_placement_state',
                                 u'keywordPlacementState'],
    u'Landing Page Title': [u'landing_page_title', u'landingPageTitle'],
    u'Location': [u'location', u'location'],
    u'Location Extension Source': [u'location_extension_source',
                                   u'locationExtensionSource'],
    u'Location type': [u'location_type', u'locationType'],
    u'Login email': [u'login_email', u'loginEmail'],
    u'Lowest position': [u'lowest_position', u'lowestPosition'],
    u'Match type': [u'match_type', u'matchType'],
    u'Max CPC source': [u'max_cpc_source', u'maxCPCSource'],
    u'Max CPM Source': [u'max_cpm_source', u'maxCPMSource'],
    u'Max. CPA (1-per-click)': [u'max_cpa1_per_click', u'maxCPA1PerClick'],
    u'Max. CPA%': [u'max_cpa', u'maxCPA'],
    u'Max. CPC': [u'max_cpc', u'maxCPC'],
    u'Max. CPM': [u'max_cpm', u'maxCPM'],
    u'Max. CPP': [u'max_cpp', u'maxCPP'],
    u'Member Count': [u'member_count', u'memberCount'],
    u'Metro area': [u'metro_area', u'metroArea'],
    u'Month': [u'month', u'month'],
    u'Month of Year': [u'month_of_year', u'monthOfYear'],
    u'Most specific location': [u'most_specific_location',
                                u'mostSpecificLocation'],
    u'Negative keyword': [u'negative_keyword', u'negativeKeyword'],
    u'Network': [u'network', u'network'],
    u'Network (with search partners)': [u'network_with_search_partners',
                                        u'networkWithSearchPartners'],
    u'PTR': [u'ptr', u'ptr'],
    u'Page': [u'page', u'page'],
    u'Phone bid type': [u'phone_bid_type', u'phoneBidType'],
    u'Phone calls': [u'phone_calls', u'phoneCalls'],
    u'Phone cost': [u'phone_cost', u'phoneCost'],
    u'Phone impressions': [u'phone_impressions', u'phoneImpressions'],
    u'Placement': [u'placement', u'placement'],
    u'Placement state': [u'placement_state', u'placementState'],
    u'Position ACE indicator': [u'position_ace_indicator',
                                u'positionACEIndicator'],
    u'Quality score': [u'quality_score', u'qualityScore'],
    u'Quarter': [u'quarter', u'quarter'],
    u'Reference Count': [u'reference_count', u'referenceCount'],
    u'Region': [u'region', u'region'],
    u'Relative CTR': [u'relative_ctr', u'relativeCTR'],
    u'Search Exact match IS': [u'search_exact_match_is', u'searchExactMatchIS'],
    u'Search Impr. share': [u'search_impr_share', u'searchImprShare'],
    u'Search Lost IS (budget)': [u'search_lost_is_budget', u'searchLostISBudget'],
    u'Search Lost IS (rank)': [u'search_lost_is_rank', u'searchLostISRank'],
    u'Search term': [u'search_term', u'searchTerm'],
    u'Second level sub-categories': [u'second_level_sub_categories',
                                     u'secondLevelSubCategories'],
    u'Shared Set ID': [u'shared_set_id', u'sharedSetID'],
    u'Shared Set Name': [u'shared_set_name', u'sharedSetName'],
    u'Shared Set Type': [u'shared_set_type', u'sharedSetType'],
    u'Start time': [u'start_time', u'startTime'],
    u'State': [u'state', u'state'],
    u'Status': [u'status', u'status'],
    u'Targeting Mode': [u'targeting_mode', u'targetingMode'],
    u'This extension vs. Other': [u'this_extension_vs_other',
                                  u'thisExtensionVsOther'],
    u'Time zone': [u'time_zone', u'timeZone'],
    u'Top level categories': [u'top_level_categories', u'topLevelCategories'],
    u'Top of page CPC': [u'top_of_page_cpc', u'topOfPageCPC'],
    u'Top vs. side': [u'top_vs_side', u'topVsSide'],
    u'Topic': [u'topic', u'topic'],
    u'Topic state': [u'topic_state', u'topicState'],
    u'Total conv. value': [u'total_conv_value', u'totalConvValue'],
    u'Total cost': [u'total_cost', u'totalCost'],
    u'URL': [u'url', u'url'],
    u'Unique Users': [u'unique_users', u'uniqueUsers'],
    u'Value / conv. (1-per-click)': [u'value_conv1_per_click',
                                     u'valueConv1PerClick'],
    u'Value / conv. (many-per-click)': [u'value_conv_many_per_click',
                                        u'valueConvManyPerClick'],
    u'View-through conv.': [u'view_through_conv', u'viewThroughConv'],
    u'View-through conv. ACE indicator': [u'view_through_conv_ace_indicator',
                                          u'viewThroughConvACEIndicator'],
    u'Week': [u'week', u'week'],
    u'Year': [u'year', u'year']}


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
    print len(df)
