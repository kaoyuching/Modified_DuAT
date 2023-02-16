# Modified DuAT
Modified DuAT uses the structure proposed by paper [DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation](https://arxiv.org/abs/2212.11677)
and replaces the GLSA module with GLAM(global-local attention module) from [All the attention you need: Global-local, spatial-channel attention for image retrieval](https://arxiv.org/abs/2107.08000).
In modified DuAT, encoder is [PVT-v2](https://github.com/whai362/PVT).


## Dependencies
- Python
    - [x] 3.8
- PyTorch
    - [x] 1.9.0
- torchvision
    - [x] 0.10.0
- timm
    - [x] 0.4.12


## Startup
### Install from a specific tag/branch
1. Install by pip

    ```shell
    $ pip install git+https://github.com/kaoyuching/Modified_DuAT.git@<tag or branch>
    ```

### Install (develop)
1. Clone form git

    ```shell
    $ git clone https://github.com/kaoyuching/Modified_DuAT.git
    cd Modified_DuAT
    ```

2. Install pytorch by yourself

3. Install requirements

    ```shell
    $ pip install -r requirements_dev.txt
    ```

4. Install `modified_duat` in repository folder

    ```shell
    $ pip install -e .
    ````


## Usage
```python
    from modified_duat.duat import DuAT

    model = DuAT(
        in_channels=3,
        encoder_name='pvt_v2_b2',
        encoder_pretrained=None,
        num_classes=3,
    )
```
