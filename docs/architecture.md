# Architecture

SetQuerry follows a fixed, observable pipeline: input query and image references are validated, routed to one of three ML adapters, formatted with evidence and confidence fields, and served through the API to the frontend. The router uses supplied modalities only; it is not a fourth ML model.
