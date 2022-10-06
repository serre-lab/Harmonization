

## **Harmonized ViT** vs **ViT**

<div class="gallery-container">

    <div class="overlay-left"> 

            <div> Click-Me (Human) </div>
            <div> Baseline model </div>
            <div> Harmonized model </div>

    </div>

    <div class="gallery" id="vit">

    </div>

</div>

## **Harmonized VGG** vs **VGG**

<div class="gallery-container">

    <div class="overlay-left">

            <div> Click-Me (Human) </div>
            <div> Baseline model </div>
            <div> Harmonized model </div>

    </div>

    <div class="gallery" id="vgg">

    </div>

</div>

## **Harmonized EfficientNetB0** vs **EfficientNetB0**

<div class="gallery-container">

    <div class="overlay-left">

            <div> Click-Me (Human) </div>
            <div> Baseline model </div>
            <div> Harmonized model </div>

    </div>

    <div class="gallery" id="effnet">

    </div>

</div>

## **Harmonized ResNet50** vs **ResNet50**

<div class="gallery-container">

    <div class="overlay-left">

            <div> Click-Me (Human) </div>
            <div> Baseline model </div>
            <div> Harmonized model </div>

    </div>

    <div class="gallery" id="resnet">

    </div>

</div>



<script defer>

window.addEventListener('DOMContentLoaded', function() {

    function is_ascendent(parent, child) {
        var node = child;
        while (node != null) {
            if (node == parent) {
                return true;
            }
            node = node.parentNode;
        }
        return false;
    }

    function event_prevent_default(e) {
        e = e || window.event;
        if (e.preventDefault)
            e.preventDefault();
        e.returnValue = false;
    }

    const NB_IMAGES = 100
    const SCROLL_POWER = 50

    const GALLERY_DATA = [
        ['vit', 'vit_baseline_faithful-wind.h5', 'vit_harmonized_solar-shadow.h5'],
        ['vgg', 'vgg16', 'vgg_frosty_eon'],
        ['effnet', 'efficientnet_b0', 'efficientnet_stellar-frog_8.h5'],
        ['resnet', 'resnet50_baseline', 'saliency_volcanic_monkey'],
    ];

    GALLERY_DATA.forEach((data) => {

        [gallery_name, model_baseline, model_harmonized] = data

        const gallery = document.getElementById(gallery_name)
        const horizontal_scroll_event = (e) => {

            if (is_ascendent(gallery, e.target)) {
                if (e.deltaY > 0) gallery.scrollLeft += SCROLL_POWER;
                else gallery.scrollLeft -= SCROLL_POWER;

                window.scrollTop -= e.wheelDeltaY;
                e.preventDefault();
                e.stopPropagation();
            }

        }

        window.addEventListener("wheel", horizontal_scroll_event, { passive: false })

        const create_single_sample = (id) => {
            return `
                <div class="single-sample">
                    <img class="explanation" loading="lazy" src="https://storage.googleapis.com/serrelab/prj_harmonization/qualitative_data/clickme/${id}.png">
                    <img class="explanation" loading="lazy" src="https://storage.googleapis.com/serrelab/prj_harmonization/qualitative_data/${model_baseline}/${id}.png">
                    <img class="explanation harmonized" loading="lazy" src="https://storage.googleapis.com/serrelab/prj_harmonization/qualitative_data/${model_harmonized}/${id}.png">
                </div>
            `
        }

        for (let i = 0; i < NB_IMAGES; i++) {
            gallery.innerHTML += create_single_sample(i)
        }

    })
});



</script>

<style>

.single-sample {
    display: inline-flex;
    flex-direction: column;
}

.gallery-container {
    position: relative;
}

.gallery {
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
    position: relative;
    padding: 20px;
    padding-left: 105px;
    box-sizing: border-box;
}

::-webkit-scrollbar {
  display: none;
}

body {
  -ms-overflow-style: none;  /* IE and Edge */
  scrollbar-width: none;  /* Firefox */
}

.explanation {
    width: 150px;
    border: solid 0px;
    background: transparent;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    border: solid 3px transparent;
    margin: 2px;
    border-radius: 5px;
}
.explanation.harmonized {
    border-color: var(--primary);
}

.overlay-left {
    position: absolute;
    left: 0;
    top:0;
    background: var(--md-default-bg-color);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    text-align: center;
    width: 100px;
    flex-wrap: wrap;
    overflow-x: hidden;
    white-space: initial;
    z-index: 2;
    font-size: 18px
}


</style>