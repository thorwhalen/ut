__author__ = 'thorwhalen'


def slurp_with_login_and_pwd():
    import sys
    import mechanize
    # sys.path.append('ClientCookie-1.0.3')
    # from mechanize import ClientCookie
    # sys.path.append('ClientForm-0.1.17')
    # import ClientForm

    # Create special URL opener (for User-Agent) and cookieJar
    cookieJar = mechanize.CookieJar()

    opener = mechanize.build_opener(mechanize.HTTPCookieProcessor(cookieJar))
    opener.addheaders = [("User-agent","Mozilla/5.0 (compatible)")]
    mechanize.install_opener(opener)
    fp = mechanize.urlopen("http://login.yahoo.com")
    forms = mechanize.ParseResponse(fp)
    fp.close()

    # print forms on this page
    for form in forms:
        print("***************************")
        print(form)

    form = forms[0]
    form["login"]  = "yahoo-user-id" # use your userid
    form["passwd"] = "password"      # use your password
    fp = mechanize.urlopen(form.click())
    fp.close()
    fp = mechanize.urlopen("https://class.coursera.org/ml-003/lecture/download.mp4?lecture_id=1") # use your group
    fp.readlines()
    fp.close()