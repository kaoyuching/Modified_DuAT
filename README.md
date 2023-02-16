# Modified DuAT
Modified DuAT uses the structure proposed by paper [DuAT: Dual-Aggregation Transformer Network for Medical Image Segmentation](https://arxiv.org/abs/2212.11677)
and replaces the GLSA module with GLAM(global-local attention module) from [All the attention you need: Global-local, spatial-channel attention for image retrieval](https://arxiv.org/abs/2107.08000).
In modified DuAT, use encoder [PVT-v2](https://github.com/whai362/PVT) as encoder.


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
1. Model input parameters:
    - `in_channels`:
        Number of channels of input image.
    - `encoder_name` should be one of name in the following:
        'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li'
    - `encoder_pretrained`:
        File path of the pretrained weight.
        Pretrained weight can get from [PVT repo](https://github.com/whai362/PVT/tree/v2/classification)
    - `num_classes`:
        Mask class number.

2. Example

```python
    from modified_duat.duat import DuAT

    model = DuAT(
        in_channels=3,
        encoder_name='pvt_v2_b2',
        encoder_pretrained=None,
        num_classes=3,
    )
```


## Acknowledgement
- DuAT: https://github.com/Barrett-python/DuAT/tree/a643560281446d71a25b47eabca1ca133bf13111
- PVT: https://github.com/whai362/PVT
- GLAM: https://arxiv.org/abs/2107.08000
