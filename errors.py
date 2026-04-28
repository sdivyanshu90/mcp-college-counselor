from __future__ import annotations


class AppError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ToolError(AppError):
    pass


class ScraperBlockedError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__(code="scraper_blocked", message=message)


class DatabaseError(AppError):
    def __init__(self, message: str, original_exception: Exception) -> None:
        super().__init__(code="database_error", message=message)
        self.original_exception = original_exception