__author__ = 'thor'



# Searching for ipod through web data:
# http://yboss.yahooapis.com/ysearch/web?q=ipod
# Search the web, images, news services with different query
# http://yboss.yahooapis.com/ysearch/web,images?web.q=ipod&images.q=mp3
# Search the web, images, news services with different queries and different SiteRestricts
# http://yboss.yahooapis.com/ysearch/limitedweb,news?limitedweb.q=google&limitedweb.sites=apple.com,techcrunch.com&news.q=mp3&news.sites=news.yahoo.com

rep = dict()
rep[' '] = '%20'
rep['"'] = '%22'

rep['/'] = '%2F'
rep['?'] = '%3F'
rep['&'] = '%26'




# http://developer.yahoo.com/boss/search/boss_api_guide/reserve_chars_esc_val.html
# Reserved character	Escape value
#
#
# semicolon (;)
# %3B
# colon (:)
# %3A
# commercial at sign (@)
# %40
# comma (,)
# %2C
# dollar sign ($)
# %24
# equals sign (=)
# %3D
#
# percent sign (%)
# %25
# quotation marks (")
# %22
# plus sign (+)
# %2B
# hash (#)
# %23
# asterisk (*)
# %2A
# less-than sign (<)
# %3C
# greater-than sign (>)
# %3E
# left brace ({)
# %7B
# right brace (})
# %7D
# vertical bar (|)
# %7C
# left square bracket ([)
# %5B
# right square bracket (])
# %5D
# circumflex (^)
# %5E
# backslash (\)
# %5C
# accent grave (`)
# %60
# left parenthesis (()	%28
# right parenthesis ())	%29