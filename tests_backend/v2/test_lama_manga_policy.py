from copy import deepcopy
import json

from sqlalchemy import update

from src.backend_v2.public_policy import DEFAULT_PUBLIC_USER_POLICY, PublicUserPolicyRepository
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.schema import metadata, platform_config
from src.backend_v2.storage.seeding import seed_system_records


def test_existing_policy_gains_manga_without_changing_other_limits(tmp_path):
    engine = create_sqlite_engine(tmp_path / "policy.sqlite3")
    try:
        metadata.create_all(engine)
        seed_system_records(engine)
        previous = deepcopy(DEFAULT_PUBLIC_USER_POLICY)
        del previous["models"]["lama_manga"]
        previous["models"]["lama_mpe"] = False
        with engine.begin() as connection:
            connection.execute(update(platform_config).values(public_user_policy_json=json.dumps(previous)))
        repository = PublicUserPolicyRepository(engine)
        upgraded = repository.load()
        assert upgraded["models"].pop("lama_manga") is True
        assert upgraded == previous
        upgraded["models"]["lama_manga"] = False
        repository.save(upgraded)
        assert repository.load()["models"]["lama_manga"] is False
    finally:
        engine.dispose()
