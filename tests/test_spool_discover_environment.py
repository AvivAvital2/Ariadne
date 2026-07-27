"""Dynamic environment discovery for `spools create`.

A shipped recipe (`spool_content/recipes/<env>.yaml`) still wins. When there is
none, `create` searches GitHub for repositories NAMED `<env>` (exact-name,
most-starred first): one match is used, several are shown for the user to pick
(or paste a URL), none prompts for a URL. Real search results — never guessed
repo URLs — with a `search_fn` seam so the flow is testable offline.
"""
from __future__ import annotations

import pytest
import yaml

import spool_acquire
from spool_acquire import SpoolError, setup_recipe


def _item(name, full_name, stars, *, language='Go', url=None, desc=''):
    return {
        'name': name,
        'full_name': full_name,
        'clone_url': url or f'https://github.com/{full_name}.git',
        'stargazers_count': stars,
        'language': language,
        'description': desc,
    }


class TestSearchRepos:
    def test_exact_name_match_ranked_by_stars(self) -> None:
        items = [
            _item('terraform', 'somefork/terraform', 5),
            _item('terraform-provider-aws', 'hashicorp/terraform-provider-aws', 9000),
            _item('terraform', 'hashicorp/terraform', 42000),
        ]
        repos = spool_acquire._search_repos('terraform', search_fn=lambda n: items)
        # exact-name only (drops terraform-provider-aws), most-starred first
        assert [r['full_name'] for r in repos] == [
            'hashicorp/terraform', 'somefork/terraform']
        assert repos[0]['url'] == 'https://github.com/hashicorp/terraform.git'
        assert repos[0]['stars'] == 42000

    def test_case_insensitive_name_match(self) -> None:
        items = [_item('Terraform', 'hashicorp/terraform', 1)]
        repos = spool_acquire._search_repos('terraform', search_fn=lambda n: items)
        assert len(repos) == 1

    def test_empty_on_search_failure(self) -> None:
        def boom(_name):
            raise RuntimeError('offline / rate-limited')
        assert spool_acquire._search_repos('x', search_fn=boom) == []


class TestDiscoverEnvironment:
    def test_single_match_used_with_language(self) -> None:
        search = lambda n: [_item('terraform', 'hashicorp/terraform', 42000)]
        base = spool_acquire._discover_environment(
            'terraform', lambda p: '', search)
        assert base['name'] == 'terraform'
        assert base['corpus']['terraform']['url'] == \
            'https://github.com/hashicorp/terraform.git'
        # GitHub's primary language → the declared spool language (grounding gate)
        assert base['languages'] == ['go']

    def test_multiple_matches_user_picks_number(self) -> None:
        matches = [_item('foo', 'a/foo', 100), _item('foo', 'b/foo', 50)]
        base = spool_acquire._discover_environment(
            'foo', lambda p: '2', lambda n: matches)
        assert base['corpus']['foo']['url'] == 'https://github.com/b/foo.git'

    def test_multiple_matches_user_pastes_url_override(self) -> None:
        matches = [_item('foo', 'a/foo', 100), _item('foo', 'b/foo', 50)]
        base = spool_acquire._discover_environment(
            'foo', lambda p: 'https://example.com/mine/foo.git',
            lambda n: matches)
        assert base['corpus']['foo']['url'] == 'https://example.com/mine/foo.git'

    def test_no_match_prompts_for_url(self) -> None:
        base = spool_acquire._discover_environment(
            'weird', lambda p: 'https://github.com/me/weird.git', lambda n: [])
        assert base['corpus']['weird']['url'] == 'https://github.com/me/weird.git'

    def test_no_match_no_url_raises(self) -> None:
        with pytest.raises(SpoolError):
            spool_acquire._discover_environment('weird', lambda p: '', lambda n: [])


class TestSetupRecipeIntegration:
    def test_discovers_repo_when_no_recipe(self, tmp_path) -> None:
        # 'gizmo' ships no recipe, so this genuinely exercises the discovery
        # path (terraform now HAS a recipe → it would take the recipe branch).
        out = tmp_path / 'gizmo.yaml'
        search = lambda n: [_item('gizmo', 'acme/gizmo', 999)]

        def _prompt(p: str) -> str:
            # pick the first version tag when the cascade asks; Enter elsewhere
            return '1' if ('tag' in p.lower() or 'pick' in p.lower()) else ''

        setup_recipe(
            'gizmo', out_path=out, available=['databricks'],
            prompt=_prompt,
            tags_fn=lambda url: ['v1.9.0', 'v1.8.0'],
            compat_fn=lambda *a, **k: {},          # no LLM compat line → offer tags
            order_fn=lambda *a, **k: ['gizmo'],
            search_fn=search,
        )
        data = yaml.safe_load(out.read_text())
        assert data['name'] == 'gizmo'
        # discovery keeps GitHub's canonical clone_url (with .git)
        assert data['corpus']['gizmo']['url'] == 'https://github.com/acme/gizmo.git'
        assert data['corpus']['gizmo']['tag'] == 'v1.9.0'
        assert data.get('languages') == ['go']
        # a discovered env has no separate runtime edition — the version IS it,
        # filled so the pack stays runtime-pinned (schema/enable gate need it).
        assert data['runtime']

    def test_opentofu_recipe_single_repo_version_is_runtime(
            self, tmp_path) -> None:
        # The shipped opentofu recipe is single-repo with no `runtime:` — the
        # corpus is OpenTofu (the MPL/open-source Terraform implementation;
        # the recipe is NAMED for its actual corpus, never BUSL
        # hashicorp/terraform). The picked version becomes the edition, and
        # no "Runtime edition" prompt is shown (nothing meaningful to type).
        # Uses the real recipe; fakes tags.
        out = tmp_path / 'tofu.yaml'
        prompts = []

        def _prompt(p: str) -> str:
            prompts.append(p)
            return '1' if ('tag' in p.lower() or 'pick' in p.lower()) else ''

        setup_recipe(
            'opentofu', out_path=out, prompt=_prompt,
            tags_fn=lambda url: ['v1.13.0', 'v1.12.0'],
            compat_fn=lambda *a, **k: {},
            order_fn=lambda *a, **k: ['opentofu'],
        )
        data = yaml.safe_load(out.read_text())
        assert 'opentofu/opentofu' in data['corpus']['opentofu']['url']
        assert data['corpus']['opentofu']['tag'] == 'v1.13.0'
        assert data['languages'] == ['go']
        assert data['runtime'] == 'v1.13.0'      # version became the edition
        assert not any('runtime edition' in p.lower() for p in prompts)

    def test_shipped_recipe_wins_and_search_is_never_called(self, tmp_path) -> None:
        called = []
        setup_recipe(
            'databricks', out_path=tmp_path / 'd.yaml', available=['databricks'],
            prompt=lambda p: '',
            tags_fn=lambda url: ['v4.0.0'],
            compat_fn=lambda *a, **k: {
                'spark': '4.0', 'delta': '4.0', 'databricks-sdk-py': '0.121'},
            order_fn=lambda *a, **k: ['spark', 'databricks-sdk-py', 'delta'],
            search_fn=lambda n: called.append(n) or [],
        )
        assert called == []
