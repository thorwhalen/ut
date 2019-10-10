__author__ = 'thorwhalen'





lu_name_x_dict = {
    'a_ce_split': ['aCESplit', 'ACE split'],
    'account': ['account', 'Account'],
    'account_id': ['accountID', 'Account ID'],
    'ad': ['ad', 'Ad'],
    'ad_approval_status': ['adApprovalStatus', 'Ad Approval Status'],
    'ad_extension_id': ['adExtensionID', 'Ad Extension ID'],
    'ad_extension_type': ['adExtensionType', 'Ad Extension Type'],
    'ad_group': ['adGroup', 'Ad group'],
    'ad_group_id': ['adGroupID', 'Ad group ID'],
    'ad_group_state': ['adGroupState', 'Ad group state'],
    'ad_id': ['adID', 'Ad ID'],
    'ad_state': ['adState', 'Ad state'],
    'ad_type': ['adType', 'Ad type'],
    'added': ['added', 'Added'],
    'approval_status': ['approvalStatus', 'Approval Status'],
    'attribute_values': ['attributeValues', 'Attribute Values'],
    'audience': ['audience', 'Audience'],
    'audience_state': ['audienceState', 'Audience state'],
    'avg_cpc': ['avgCPC', 'Avg. CPC'],
    'avg_cpm': ['avgCPM', 'Avg. CPM'],
    'avg_cpp': ['avgCPP', 'Avg. CPP'],
    'avg_position': ['avgPosition', 'Avg. position'],
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
    'campaign_id': ['campaignID', 'Campaign ID'],
    'campaign_state': ['campaignState', 'Campaign state'],
    'campaigns': ['campaigns', '# Campaigns'],
    'categories': ['categories', 'Categories'],
    'city': ['city', 'City'],
    'click_id': ['clickId', 'Click Id'],
    'click_type': ['clickType', 'Click type'],
    'clicks': ['clicks', 'Clicks'],
    'clicks_ace_indicator': ['clicksACEIndicator', 'Clicks ACE indicator'],
    'client_name': ['clientName', 'Client name'],
    'company_name': ['companyName', 'Company name'],
    'content_impr_share': ['contentImprShare', 'Content Impr. share'],
    'content_lost_is_budget': ['contentLostISBudget',
                                'Content Lost IS (budget)'],
    'content_lost_is_rank': ['contentLostISRank', 'Content Lost IS (rank)'],
    'conv': ['conv', 'Conv.'],
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
    'cost': ['cost', 'Cost'],
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
    'criterion_id': ['criterionID', 'Criterion ID'],
    'ctr': ['ctr', 'CTR'],
    'currency': ['currency', 'Currency'],
    'customer_id': ['customerID', 'Customer ID'],
    'day': ['day', 'Day'],
    'day_of_week': ['dayOfWeek', 'Day of week'],
    'default_max_cpc': ['defaultMaxCPC', 'Default max. CPC'],
    'delivery_method': ['deliveryMethod', 'Delivery method'],
    'description_line1': ['descriptionLine1', 'Description line 1'],
    'description_line2': ['descriptionLine2', 'Description line 2'],
    'destination_url': ['destinationURL', 'Destination URL'],
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
    'impressions': ['impressions', 'Impressions'],
    'impressions_ace_indicator': ['impressionsACEIndicator',
                                   'Impressions ACE indicator'],
    'invalid_click_rate': ['invalidClickRate', 'Invalid click rate'],
    'invalid_clicks': ['invalidClicks', 'Invalid clicks'],
    'is_negative': ['isNegative', 'Is negative'],
    'is_targetable': ['isTargetable', 'Is Targetable'],
    'keyword': ['keyword', 'Keyword'],
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
    'match_type': ['matchType', 'Match type'],
    'max_cpa': ['maxCPA', 'Max. CPA%'],
    'max_cpa1_per_click': ['maxCPA1PerClick', 'Max. CPA (1-per-click)'],
    'max_cpc': ['maxCPC', 'Max. CPC'],
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
    'value_conv1_per_click': ['valueConv1PerClick',
                               'Value / conv. (1-per-click)'],
    'value_conv_many_per_click': ['valueConvManyPerClick',
                                   'Value / conv. (many-per-click)'],
    'view_through_conv': ['viewThroughConv', 'View-through conv.'],
    'view_through_conv_ace_indicator': ['viewThroughConvACEIndicator',
                                         'View-through conv. ACE indicator'],
    'week': ['week', 'Week'],
    'year': ['year', 'Year']}