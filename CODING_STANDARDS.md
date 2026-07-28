# Coding Standards & Conventions

## Python

### Style
- **Formatter**: Ruff (line length 100, double quotes)
- **Linter**: Ruff with selected rulesets (E, F, I, N, W, UP, ANN, B, C4, SIM, TID)
- **Type Checker**: mypy (strict mode)
- Run `make lint`, `make format`, `make typecheck` before committing

### Naming
| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `FoodClassifier` |
| Functions/Methods | snake_case | `predict_food_type()` |
| Variables | snake_case | `freshness_score` |
| Constants | UPPER_SNAKE | `FOOD_CATEGORIES` |
| Private members | _prefix | `_load_model()` |
| Type vars | PascalCase | `T` |

### Imports (sorted by Ruff)
1. Standard library
2. Third-party
3. Local (absolute imports preferred)

### Architecture
- **Clean Architecture**: domain → core → infrastructure → api
- **No circular imports** between layers
- Domain layer has zero external dependencies
- Use Pydantic for schema validation
- Use SQLAlchemy for ORM

## TypeScript / React

### Style
- **Formatted with**: Prettier (included in Next.js config)
- **Strict TypeScript**: `strict: true` in tsconfig.json

### Naming
- Components: PascalCase (`InspectionCard`)
- Hooks: camelCase with `use` prefix (`useAuth`)
- Files: PascalCase for components, camelCase for utilities
- Props interfaces: `<ComponentName>Props`

### Conventions
- Functional components with hooks
- Tailwind CSS for styling
- Single-responsibility components
- Props typed with TypeScript interfaces

## Git

### Commit Messages
```
type(scope): brief description

- feat: new feature
- fix: bug fix
- refactor: code restructuring
- docs: documentation
- style: formatting
- test: testing
- chore: maintenance
```

### Branching
- `main` — production-ready
- `feat/<name>` — new features
- `fix/<name>` — bug fixes
- Always branch from `main`, PR back to `main`

## Pre-commit Hooks

Before every commit:
1. `ruff check .` — lint
2. `ruff format . --check` — format check
3. `mypy backend/ ai-engine/` — type check
4. `pytest` — run tests
