#undef NDEBUG
#include <assert.h>
#include "libimagequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char magic[] = "magic";

static void test_log_callback_function(const liq_attr *at, const char *message, void* user_info) {
    assert(at);
    assert(user_info == magic);
    assert(message);
    assert(strlen(message));
}

static int test_abort_callback(float progress_percent, void* user_info) {
    assert(progress_percent >= 0.0 && progress_percent <= 100.0);
    assert(user_info == magic);
    return 0;
}

static int progress_called = 0;
static int test_continue_callback(float progress_percent, void* user_info) {
    assert(progress_percent >= 0.0 && progress_percent <= 100.0);
    assert(user_info == magic);
    progress_called++;
    return 1;
}

static void test_abort() {
    liq_attr *attr = liq_attr_create();

    unsigned char dummy[4] = {0};
    liq_image *img = liq_image_create_rgba(attr, dummy, 1, 1, 0);

    liq_attr_set_progress_callback(attr, test_abort_callback, magic);

    liq_result *res = liq_quantize_image(attr, img);
    assert(!res);

    liq_attr_destroy(attr);
}

static void test_zero_histogram() {
    liq_attr *attr = liq_attr_create();
    liq_histogram *hist = liq_histogram_create(attr);
    assert(hist);

    liq_result *res;
    liq_error err = liq_histogram_quantize(hist, attr, &res);
    assert(!res);
    assert(err);

    liq_attr_destroy(attr);
    liq_histogram_destroy(hist);
}

static void test_histogram() {
    liq_attr *attr = liq_attr_create();
    liq_histogram *hist = liq_histogram_create(attr);
    assert(hist);

    const unsigned char dummy1[4] = {255,0,255,255};
    liq_image *const img1 = liq_image_create_rgba(attr, dummy1, 1, 1, 0);
    assert(img1);

    const liq_error err1 = liq_histogram_add_image(hist, attr, img1);
    assert(LIQ_OK == err1);

    const unsigned char dummy2[4] = {0,0,0,0};
    liq_image *const img2 = liq_image_create_rgba(attr, dummy2, 1, 1, 0);
    assert(img2);
    liq_image_add_fixed_color(img2, (liq_color){255,255,255,255});


    const liq_error err2 = liq_histogram_add_image(hist, attr, img2);
    assert(LIQ_OK == err2);

    liq_image_destroy(img1);
    liq_image_destroy(img2);

    liq_result *res;
    liq_error err = liq_histogram_quantize(hist, attr, &res);
    assert(LIQ_OK == err);
    assert(res);

    liq_attr_destroy(attr);

    liq_histogram_destroy(hist);

    const liq_palette *pal = liq_get_palette(res);
    assert(pal);
    assert(pal->count == 3);

    liq_result_destroy(res);
}

static void test_fixed_colors() {
    liq_attr *attr = liq_attr_create();

    liq_attr_set_progress_callback(attr, test_continue_callback, magic);
    liq_set_log_callback(attr, test_log_callback_function, magic);

    unsigned char dummy[4] = {0};
    liq_image *img = liq_image_create_rgba(attr, dummy, 1, 1, 0);
    assert(img);

    liq_image_add_fixed_color(img, (liq_color){0,0,0,0});

    liq_result *res = liq_quantize_image(attr, img);
    assert(res);
    assert(progress_called);

    const liq_palette *pal = liq_get_palette(res);
    assert(pal);
    assert(pal->count == 1);

    liq_result_destroy(res);
    liq_image_destroy(img);
    liq_attr_destroy(attr);
}

static void test_fixed_colors_order() {
    liq_attr *attr = liq_attr_create();

    unsigned char dummy[4] = {0};
    liq_image *img = liq_image_create_rgba(attr, dummy, 1, 1, 0);
    assert(img);

    liq_color colors[17] = {
        {0,0,0,0}, {1,1,1,1}, {2,2,2,2}, {3,3,3,3}, {4,4,4,4}, {5,4,4,4},
        {6,4,4,4}, {6,7,4,4}, {6,7,8,4}, {6,7,8,9}, {10,7,8,9}, {10,11,8,9},
        {10,11,12,9}, {10,11,12,13}, {14,11,12,13}, {14,15,12,13}, {253,254,255,254},
    };

    for(int i=0; i < 17; i++) {
        liq_image_add_fixed_color(img, colors[i]);
    }

    liq_result *res = liq_quantize_image(attr, img);
    assert(res);

    const liq_palette *pal = liq_get_palette(res);
    assert(pal);
    assert(pal->count == 17);

    for(int i=0; i < 17; i++) {
        assert(pal->entries[i].r == colors[i].r);
        assert(pal->entries[i].g == colors[i].g);
        assert(pal->entries[i].b == colors[i].b);
        assert(pal->entries[i].a == colors[i].a);
    }

    liq_set_dithering_level(res, 1.0);

    char buf[1];
    assert(LIQ_OK == liq_write_remapped_image(res, img, buf, 1));

    liq_result_set_progress_callback(res, test_abort_callback, magic);
    assert(LIQ_ABORTED == liq_write_remapped_image(res, img, buf, 1));

    liq_result_destroy(res);
    liq_image_destroy(img);
    liq_attr_destroy(attr);
}

static void test_16bit_inputs(void) {
    liq_attr *attr = liq_attr_create();
    assert(attr);

    const int width = 8;
    const int height = 4;
    const int count = width * height;

    // 1. contiguous 16-bit bitmap
    liq_color16 *bitmap = malloc((size_t)count * sizeof(liq_color16));
    assert(bitmap);
    for (int i = 0; i < count; i++) {
        unsigned short v = (unsigned short)(i * 257);
        bitmap[i] = (liq_color16){ v, (unsigned short)(v / 2), (unsigned short)(65535 - v), 65535 };
    }
    liq_image *img = liq_image_create_rgba16(attr, bitmap, width, height, 0);
    assert(img);

    liq_result *res = liq_quantize_image(attr, img);
    assert(res);
    const liq_palette *pal = liq_get_palette(res);
    assert(pal);
    assert(pal->count > 0 && pal->count <= 256);

    unsigned char *out = malloc((size_t)count);
    assert(out);
    liq_set_dithering_level(res, 1.0);
    assert(LIQ_OK == liq_write_remapped_image(res, img, out, (size_t)count));
    liq_result_destroy(res);
    liq_image_destroy(img);
    free(out);

    // 2. per-row 16-bit pointers
    void **rows = malloc((size_t)height * sizeof(void*));
    assert(rows);
    for (int y = 0; y < height; y++) {
        rows[y] = (void*)&bitmap[y * width];
    }
    img = liq_image_create_rgba16_rows(attr, rows, width, height, 0);
    assert(img);
    res = liq_quantize_image(attr, img);
    assert(res);
    pal = liq_get_palette(res);
    assert(pal && pal->count > 0);
    liq_result_destroy(res);
    liq_image_destroy(img);
    free(rows);
    free(bitmap);

    liq_attr_destroy(attr);
}

static int custom16_calls = 0;
static void get_rgba16_row(liq_color16 row_out[], int row, int width, void *user_info) {
    assert(user_info == magic);
    assert(width == 6);
    assert(row >= 0 && row < 3);
    for (int i = 0; i < width; i++) {
        unsigned short v = (unsigned short)((i * 1000 + row * 333) & 0xffff);
        row_out[i] = (liq_color16){ v, v, v, 65535 };
    }
    custom16_calls++;
}

static void test_16bit_custom_callback(void) {
    liq_attr *attr = liq_attr_create();
    assert(attr);
    liq_image *img = liq_image_create_custom16(attr, get_rgba16_row, magic, 6, 3, 0);
    assert(img);
    liq_result *res = liq_quantize_image(attr, img);
    assert(res);
    const liq_palette *pal = liq_get_palette(res);
    assert(pal && pal->count > 0);
    assert(custom16_calls >= 3);
    liq_result_destroy(res);
    liq_image_destroy(img);
    liq_attr_destroy(attr);
}

void run_liq_tests() {
    test_fixed_colors();
    test_fixed_colors_order();
    test_abort();
    test_histogram();
    test_zero_histogram();
    test_16bit_inputs();
    test_16bit_custom_callback();
}
