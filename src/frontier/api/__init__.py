"""HTTP surface: routers, Pydantic contracts, and the app factory.

Nothing here holds business logic. Routers translate between the wire format in
``frontier.api.schemas`` and the adapters in ``frontier.services``; everything
numerical lives further down, in ``frontier.data``, ``frontier.forecasting``,
``frontier.optimization`` and ``frontier.portfolio``.
"""
