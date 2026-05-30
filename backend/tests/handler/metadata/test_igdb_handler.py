"""Tests for the IGDB metadata handler."""

from unittest.mock import AsyncMock, patch

import pytest

from adapters.services.igdb_types import GameType
from handler.metadata.igdb_handler import IGDBHandler
from models.rom import Rom

GENESIS_IGDB_ID = 29


def _make_game(game_id: int, name: str) -> dict:
    """Build a minimal IGDB Game dict for testing."""
    return {
        "id": game_id,
        "name": name,
        "slug": name.lower().replace(" ", "-"),
        "summary": "",
        "total_rating": 0.0,
        "aggregated_rating": 0.0,
        "first_release_date": None,
        "artworks": [],
        "cover": None,
        "screenshots": [],
        "platforms": [{"id": GENESIS_IGDB_ID, "name": "Sega Mega Drive/Genesis"}],
        "alternative_names": [],
        "genres": [],
        "franchise": None,
        "franchises": [],
        "collections": [],
        "game_modes": [],
        "involved_companies": [],
        "expansions": [],
        "dlcs": [],
        "remasters": [],
        "remakes": [],
        "expanded_games": [],
        "ports": [],
        "similar_games": [],
        "videos": [],
        "age_ratings": [],
        "multiplayer_modes": [],
        "game_localizations": [],
    }


class TestSearchRomGameTypeFilter:
    """Tests for _search_rom game_type filtering."""

    @pytest.mark.asyncio
    async def test_standalone_expansion_included_in_game_type_filter(self):
        """Searching with game_type filter must include STANDALONE_EXPANSION
        so that games like 'Ecco: The Tides of Time' are found on the first
        search pass and not confused with their parent game."""
        handler = IGDBHandler()

        ecco_dolphin = _make_game(1799, "Ecco the Dolphin")
        ecco_tides = _make_game(5379, "Ecco: The Tides of Time")

        async def mock_list_games(
            search_term=None, fields=None, where=None, limit=None
        ):
            # First call (with game_type filter): return both games
            if where and "game_type" in where:
                # Verify STANDALONE_EXPANSION (4) is in the filter
                assert (
                    str(int(GameType.STANDALONE_EXPANSION)) in where
                ), f"STANDALONE_EXPANSION should be in game_type filter, got: {where}"
                # Simulate IGDB returning both games when the search includes
                # standalone expansions
                if search_term and "tides of time" in search_term.lower():
                    return [ecco_dolphin, ecco_tides]
                return [ecco_dolphin]
            return []

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(
                handler.igdb_service,
                "list_games",
                side_effect=mock_list_games,
            ),
            patch.object(
                handler.igdb_service,
                "search",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            result = await handler._search_rom(
                "ecco the tides of time", GENESIS_IGDB_ID, with_game_type=True
            )

        assert result is not None
        assert (
            result["id"] == 5379
        ), f"Expected Ecco: The Tides of Time (id=5379), got {result.get('name')} (id={result.get('id')})"

    @pytest.mark.asyncio
    async def test_expanded_search_uses_all_results_not_just_first(self):
        """When the primary search fails and the expanded IGDB search endpoint
        is used, all unique game IDs from the results must be fetched and
        the best match selected — not just the first result."""
        handler = IGDBHandler()

        ecco_dolphin = _make_game(1799, "Ecco the Dolphin")
        ecco_tides = _make_game(5379, "Ecco: The Tides of Time")

        # Primary search returns nothing useful
        async def mock_list_games(
            search_term=None, fields=None, where=None, limit=None
        ):
            if where and "game_type" not in where and not where.startswith("("):
                # Primary search pass — return no results so we fall through to
                # the expanded search
                return []
            if where and where.startswith("("):
                # Expanded game details lookup — return both candidates
                return [ecco_dolphin, ecco_tides]
            return []

        # Expanded search returns two results — wrong game FIRST, correct game second
        expanded_results = [
            {"game": {"id": 1799}, "name": "Ecco the Dolphin"},
            {"game": {"id": 5379}, "name": "Ecco: The Tides of Time"},
        ]

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(
                handler.igdb_service,
                "list_games",
                side_effect=mock_list_games,
            ),
            patch.object(
                handler.igdb_service,
                "search",
                new_callable=AsyncMock,
                return_value=expanded_results,
            ),
        ):
            result = await handler._search_rom(
                "ecco the tides of time", GENESIS_IGDB_ID, with_game_type=False
            )

        assert result is not None
        assert result["id"] == 5379, (
            f"Expected Ecco: The Tides of Time (id=5379), got {result.get('name')} (id={result.get('id')}). "
            "The expanded search must consider ALL results, not just the first."
        )

    @pytest.mark.asyncio
    async def test_expanded_search_matches_on_alternative_name(self):
        """When the filename uses a localized alternative name (e.g. German title),
        the expanded search must score against alternative_names — not only the
        primary English name — so the game is not discarded by find_best_match."""
        PSX_IGDB_ID = 7
        handler = IGDBHandler()

        # Primary English name: "James Bond 007: The World Is Not Enough"
        # German alternative name: "007 - Die Welt Ist Nicht Genug"
        bond_game = {
            **_make_game(158962, "James Bond 007: The World Is Not Enough"),
            "alternative_names": [
                {"name": "007 - Die Welt Ist Nicht Genug", "comment": "German title"}
            ],
        }

        async def mock_list_games(
            search_term=None, fields=None, where=None, limit=None
        ):
            if where and where.startswith("("):
                return [bond_game]
            return []

        expanded_results = [
            {"game": {"id": 158962}, "name": "007 - Die Welt Ist Nicht Genug"}
        ]

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(
                handler.igdb_service, "list_games", side_effect=mock_list_games
            ),
            patch.object(
                handler.igdb_service,
                "search",
                new_callable=AsyncMock,
                return_value=expanded_results,
            ),
        ):
            result = await handler._search_rom(
                "007 - die welt ist nicht genug", PSX_IGDB_ID, with_game_type=False
            )

        assert result is not None
        assert result["id"] == 158962, (
            f"Expected Bond game (id=158962) matched via alternative name, "
            f"got {result.get('name')} (id={result.get('id')})"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "search_term",
        [
            "007 die welt ist nicht genug",  # no dash
            "007: die welt ist nicht genug",  # colon variant
            "007 - die welt ist nicht genug",  # dash variant
        ],
    )
    async def test_primary_search_matches_via_alternative_name(self, search_term):
        """Primary search must score against alternative_names so all punctuation
        variants of a localized filename can match via the full-text search path."""
        PSX_IGDB_ID = 7
        handler = IGDBHandler()

        bond_game = {
            **_make_game(158962, "James Bond 007: The World Is Not Enough"),
            "alternative_names": [
                {"name": "007 - Die Welt Ist Nicht Genug", "comment": "German title"}
            ],
        }

        async def mock_list_games(
            search_term=None, fields=None, where=None, limit=None
        ):
            if not where or where.startswith("("):
                return []
            return [bond_game]

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(
                handler.igdb_service, "list_games", side_effect=mock_list_games
            ),
            patch.object(
                handler.igdb_service, "search", new_callable=AsyncMock, return_value=[]
            ),
        ):
            result = await handler._search_rom(
                search_term, PSX_IGDB_ID, with_game_type=False
            )

        assert (
            result is not None
        ), f"Expected match via alternative name for search_term={search_term!r}, got None"
        assert result["id"] == 158962

    @pytest.mark.asyncio
    async def test_search_term_preserves_dashes_for_igdb_wildcard(self):
        """normalize_search_term must preserve dashes so IGDB wildcard queries
        can match alternative_names stored with dashes (e.g. '007 - Die Welt...')."""

        from handler.metadata.igdb_handler import IGDBHandler

        PSX_IGDB_ID = 7
        handler = IGDBHandler()

        captured_where: list[str] = []

        bond_game = {
            **_make_game(158962, "James Bond 007: The World Is Not Enough"),
            "alternative_names": [
                {"name": "007 - Die Welt Ist Nicht Genug", "comment": "German title"}
            ],
        }

        async def mock_list_games(
            search_term=None, fields=None, where=None, limit=None
        ):
            if where and where.startswith("("):
                return [bond_game]
            return []

        async def mock_search(fields=None, where=None, limit=None):
            if where:
                captured_where.append(where)
            return [{"game": {"id": 158962}, "name": "007 - Die Welt Ist Nicht Genug"}]

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(
                handler.igdb_service, "list_games", side_effect=mock_list_games
            ),
            patch.object(handler.igdb_service, "search", side_effect=mock_search),
        ):
            await handler._search_rom(
                "007 - die welt ist nicht genug", PSX_IGDB_ID, with_game_type=False
            )

        assert captured_where, "search endpoint was never called"
        assert any(
            "-" in w for w in captured_where
        ), f"Expected dashes preserved in IGDB wildcard query, got: {captured_where}"

    @pytest.mark.asyncio
    async def test_get_rom_preserves_punctuation_in_search_term(self):
        """get_rom must normalize the term with remove_punctuation=False so
        dashes/colons reach _search_rom. The _search_rom-level tests pass the
        term in directly, so this is the only test that pins that line."""
        handler = IGDBHandler()
        PSX_IGDB_ID = 7
        captured_terms: list[str] = []

        async def fake_search_rom(search_term, platform_igdb_id, with_game_type=False):
            captured_terms.append(search_term)
            return None

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(handler, "_search_rom", side_effect=fake_search_rom),
        ):
            await handler.get_rom(
                Rom(), "007 - Die Welt Ist Nicht Genug.chd", PSX_IGDB_ID
            )

        assert captured_terms, "_search_rom was never called"
        assert all("-" in term for term in captured_terms), (
            "Expected the dash to survive normalization into the term passed to "
            f"_search_rom, got {captured_terms}"
        )

    @pytest.mark.asyncio
    async def test_quotes_are_stripped_from_igdb_query(self):
        """A double-quote in the term must not leak into the APIcalypse query,
        where it would terminate the quoted string literal and break the search.
        Dashes/colons are kept; only quotes/backslashes are removed."""
        handler = IGDBHandler()
        list_terms: list[str] = []
        search_wheres: list[str] = []

        async def mock_list_games(
            search_term=None, fields=None, where=None, limit=None
        ):
            if search_term is not None:
                list_terms.append(search_term)
            return []

        async def mock_search(fields=None, where=None, limit=None):
            if where:
                search_wheres.append(where)
            return []

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(
                handler.igdb_service, "list_games", side_effect=mock_list_games
            ),
            patch.object(handler.igdb_service, "search", side_effect=mock_search),
        ):
            await handler._search_rom(
                '007 - "die welt"', GENESIS_IGDB_ID, with_game_type=False
            )

        assert all(
            '"' not in term for term in list_terms
        ), f"Quote leaked into the search term sent to IGDB: {list_terms}"
        assert search_wheres, "expanded search endpoint was never reached"
        for where in search_wheres:
            # Two `*"..."*` wildcards => exactly four double-quotes; a stray
            # quote from the term would push this higher and break the query.
            assert where.count('"') == 4, f"stray quote leaked into where: {where}"
            assert "-" in where, f"dash should be preserved in where: {where}"

    @pytest.mark.asyncio
    async def test_name_collision_prefers_lowest_igdb_id(self):
        """When two games share a name, the lower IGDB id wins deterministically,
        independent of the order IGDB returns them in."""
        handler = IGDBHandler()

        # Same name, returned highest-id-first to prove order doesn't decide it.
        higher = _make_game(999, "Double Dragon")
        lower = _make_game(42, "Double Dragon")

        async def mock_list_games(
            search_term=None, fields=None, where=None, limit=None
        ):
            if where and "game_type" in where:
                return [higher, lower]
            return []

        with (
            patch(
                "handler.metadata.igdb_handler.IGDBHandler.is_enabled",
                return_value=True,
            ),
            patch.object(
                handler.igdb_service, "list_games", side_effect=mock_list_games
            ),
            patch.object(
                handler.igdb_service, "search", new_callable=AsyncMock, return_value=[]
            ),
        ):
            result = await handler._search_rom(
                "double dragon", GENESIS_IGDB_ID, with_game_type=True
            )

        assert result is not None
        assert (
            result["id"] == 42
        ), f"Expected the lower IGDB id (42) on a name collision, got {result['id']}"
