"""Crash-window decisions for staging-to-object publication."""

from __future__ import annotations

from enum import StrEnum


class JournalState(StrEnum):
    STAGED = "staged"
    FILE_PUBLISHED = "file_published"
    DATABASE_COMMITTED = "database_committed"


class RecoveryAction(StrEnum):
    PUBLISH_STAGING_FILE = "publish_staging_file"
    CONFIRM_AND_DELETE_JOURNAL = "confirm_and_delete_journal"
    KEEP_UNTIL_GRACE_PERIOD = "keep_until_grace_period"
    DELETE_ORPHAN_STAGING = "delete_orphan_staging"
    MARK_ASSET_MISSING = "mark_asset_missing"


def next_journal_state(current: JournalState) -> JournalState:
    if current is JournalState.STAGED:
        return JournalState.FILE_PUBLISHED
    if current is JournalState.FILE_PUBLISHED:
        return JournalState.DATABASE_COMMITTED
    raise ValueError("database_committed is terminal; delete the journal row")


def recovery_action(
    *,
    database_has_asset: bool,
    final_file_exists: bool,
    staging_file_exists: bool,
) -> RecoveryAction:
    if database_has_asset and final_file_exists:
        return RecoveryAction.CONFIRM_AND_DELETE_JOURNAL
    if database_has_asset and not final_file_exists:
        return RecoveryAction.MARK_ASSET_MISSING
    if final_file_exists:
        return RecoveryAction.KEEP_UNTIL_GRACE_PERIOD
    if staging_file_exists:
        return RecoveryAction.PUBLISH_STAGING_FILE
    return RecoveryAction.DELETE_ORPHAN_STAGING
