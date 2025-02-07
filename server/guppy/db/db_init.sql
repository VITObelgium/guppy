CREATE TABLE "layer_metadata" (
  "id" SERIAL PRIMARY KEY,
  "layer_name" varchar UNIQUE NOT NULL,
"label" varchar NOT NULL,
  "file_path" varchar NOT NULL,
  "data_path" varchar,
  "is_rgb" boolean NOT NULL,
  "rgb_factor" float,
  "is_mbtile" boolean NOT NULL,
    "metafdata_str" varchar
);
 CREATE TABLE "tile_statistics" (
   id SERIAL PRIMARY KEY,
   layer_name VARCHAR NOT NULL,
   x INTEGER NOT NULL,
   y INTEGER NOT NULL,
   z INTEGER NOT NULL,
   count INTEGER NOT NULL
 );

GRANT ALL ON all tables IN SCHEMA guppy TO guppy;
GRANT ALL ON all sequences IN SCHEMA guppy TO guppy;

