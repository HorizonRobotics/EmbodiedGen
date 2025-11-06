document.addEventListener('DOMContentLoaded', function () {

  const swiperElement = document.querySelector('.swiper1');

  if (swiperElement) {
    const swiper = new Swiper('.swiper1', {
      loop: true,
      slidesPerView: 3,
      spaceBetween: 20,
      navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
      },
      centeredSlides: false,
      noSwiping: true,
      noSwipingClass: 'swiper-no-swiping',
      watchSlidesProgress: true,
    });

    const modelViewers = swiperElement.querySelectorAll('model-viewer');

    if (modelViewers.length > 0) {
      let loadedCount = 0;
      modelViewers.forEach(mv => {
        mv.addEventListener('load', () => {
          loadedCount++;
          if (loadedCount === modelViewers.length) {
            swiper.update();
          }
        });
      });
    }
  }
});