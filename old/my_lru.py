from __future__ import division


raise DeprecationWarning("Deprecating because realized this pattern had a name, and code for it "
                         "(is even integrated in the standard libraries of Python 3+. It's called lru_cache.")

class CachedObjAccess(object):
    """
    This class allows you to not have to re-construct objects every time you need them.
     It works by caching the objects you've asked for before, and giving you the already constructed objects from
     the cache itself.
    It's basically a constructor memoizer that (1) has a limited cache and (2) gets rid of the cache entry that
    was accessed the longest time ago if the cache is at capacity when a new object needs to be inserted.

    https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_Recently_Used_.28LRU.29

    """
    def __init__(self,
                 obj_constructor,
                 constructor_kwargs_for_key,
                 cache_size = 10,
                 ckey_from_key=lambda x: x):
        """

        :param obj_constructor: The function (or class constructor) that constructs the objects.
        :param constructor_kwargs_for_key: A function that
        :param cache_size:
        :param ckey_from_key:
        """
        self.obj_constructor = obj_constructor
        self.constructor_kwargs_for_key = constructor_kwargs_for_key
        self.cache_size = cache_size
        self.ckey_from_key = ckey_from_key
        self.cobj_for_ckey = dict()  # the cache!

    def get_obj(self, key):
        """
        Get the obj for that key, if it's in the cache (cobj_for_ckey), and if it's not,
        construct it (with obj_constructor(**constructor_kwargs_for_key(key)) and insert it in the cache,
        removing the oldest (accessed the longest time ago) entry if the cache is at capacity
        (i.e. len(cobj_for_ckey) >= cache_size
        :param key:
        :return:
        """
        ckey = self.ckey_from_key(key)
        cobj = self.cobj_for_ckey.get(ckey, None)
        if cobj is not None:
            cobj['last_accessed_utc'] = datetime.utcnow()
            return cobj['obj']
        else:
            cobj = {'obj': self.obj_constructor(**self.constructor_kwargs_for_key(key)),
                    'last_accessed_utc': datetime.utcnow()}
            self.insert_cobj(ckey, cobj)
            return cobj['obj']

    def insert_cobj(self, ckey, cobj):
        if len(self.cobj_for_ckey) >= self.cache_size:  # if we're at the limit, make some space...
            # find the ckey with the minimum last_accessed_utc (the ckey that's been accessed the longest time ago)
            min_last_accessed_utc = cobj['last_accessed_utc']
            pop_ckey = None
            for _ckey, _cobj in self.cobj_for_ckey.iteritems():
                if _cobj['last_accessed_utc'] < min_last_accessed_utc:
                    min_last_accessed_utc = _cobj['last_accessed_utc']
                    pop_ckey = _ckey

            # pop the ckey that's been accessed the longest time ago
            if pop_ckey is not None:
                self.cobj_for_ckey.pop(pop_ckey)

        # add the {ckey: cobj} entry
        self.cobj_for_ckey[ckey] = cobj

