"""Supabase utils stub — implement when Supabase is needed."""


class _StorageBucket:
    def upload(self, path, *args, **kwargs):
        raise NotImplementedError("Supabase not configured")

    def get_public_url(self, path):
        raise NotImplementedError("Supabase not configured")


class _Storage:
    def from_(self, bucket):
        return _StorageBucket()

    def create_bucket(self, name, **kwargs):
        pass


class _TableQuery:
    def __init__(self):
        self._chain = self

    def eq(self, col, val):
        return self._chain

    def execute(self):
        return type("Result", (), {"data": [], "count": 0})()

    def update(self, data):
        return self._chain


class _Client:
    storage = _Storage()

    def table(self, name):
        return _TableQuery()


class SupabaseManager:
    def __init__(self):
        self._client = _Client()

    @property
    def client(self):
        return self._client

    def is_connected(self):
        return False


supabase_manager = SupabaseManager()
