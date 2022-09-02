CREATE TABLE "layer_metadata" (
  "id" SERIAL PRIMARY KEY,
  "layer_name" varchar UNIQUE NOT NULL,
  "file_path" varchar NOT NULL,
  "is_rgb" boolean NOT NULL,
  "rgb_factor" float
);

GRANT ALL ON all tables IN SCHEMA guppy2 TO guppy2;
GRANT ALL ON all sequences IN SCHEMA guppy2 TO guppy2;

