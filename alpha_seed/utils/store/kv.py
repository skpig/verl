import ray


class KVDB:

    def get(self, key: str, default=None):
        """
        get key if exists, otherwise return default
        """
        raise NotImplementedError()

    def put(self, key: str, value):
        """
        put key-value pair. evict previous if exists
        """
        raise NotImplementedError()

    def delete(self, key: str):
        """
        delete key if exists
        """
        raise NotImplementedError()

    def __len__(self):
        """
        get number of keys
        """
        raise NotImplementedError()


class InMemory(KVDB):

    def __init__(self):
        self._map = {}

    def get(self, key: str, default=None):
        return self._map.get(key, default)

    def put(self, key: str, value):
        self._map[key] = value

    def delete(self, key: str):
        self._map.pop(key, None)

    def __len__(self):
        return len(self._map)


class RayObjectKV(KVDB):

    def __init__(self):
        self.key_refs = {}  # {key -> ref}

    def get(self, key: str, default=None):
        ref = self.key_refs.get(key)
        if ref is None:
            return default
        try:
            return ray.get(ref)
        except Exception as e:
            return default

    def put(self, key, value):
        ref = ray.put(value)
        self.key_refs[key] = ref

    def delete(self, key):
        self.key_refs.pop(key, None)

    def __len__(self):
        return len(self.key_refs)
