curl -X POST -H "Content-Type: application/json" --data '{
    "repo_name": "seed/alpha-seed", 
    "cluster_label": "NVIDIA_H20|hub.byted.org/reckon/data.reckon.mlx.image_5940:26def500fbbd577bdfd57f100f6aa993"
}' 'http://data-seed-mariana-ci.byted.org/api/v1/arnold_ci/add_runner'


curl -X POST -H "Content-Type: application/json" --data '{
    "repo_name": "seed/alpha-seed", 
    "cluster_label": "NVIDIA_L20_16|hub.byted.org/reckon/data.reckon.mlx.image_5940:26def500fbbd577bdfd57f100f6aa993"
}' 'http://data-seed-mariana-ci.byted.org/api/v1/arnold_ci/add_runner'