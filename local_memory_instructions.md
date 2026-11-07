Local-first long-term memory for {user_name}.

Recall is progressive.
`memory_search` returns short previews on purpose: run it
first, then call `memory_get` with an id to read the full record. Do not paste
whole bodies into context until you have decided you need them.

`memory_store` deduplicates automatically - restating something you already
stored reinforces the existing record instead of creating a near-duplicate, and
that is a success, not an error. To replace a record whose content is now
wrong, pass `supersedes_id` rather than storing a contradictory second copy.

`memory_forget` only requests a deletion. The record leaves search results
immediately but is not destroyed until {user_name} confirms it, so never tell the user
something has been deleted on the strength of a `memory_forget` call alone.

Scopes are user-configurable. Omit `scope` on `memory_store` to use the default;
an unknown scope is rejected with the list of available ones, and `memory_stats`
lists them too.

Kind is an open vocabulary. The seeded kinds are: {seed_kinds}. Coin a new kind if none of them fit; it will be registered automatically.
