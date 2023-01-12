#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <ctime>
#include <iostream>
#include <cmath>


using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)


//Cтруктура вектора
struct vec3 {
    float x;
    float y;
    float z;
};

__device__ __host__ float dot(vec3 a, vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ vec3 prod(vec3 a, vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,  a.x * b.y - a.y * b.x};
}

__device__ __host__ vec3 norm(vec3 v) {
    float l = sqrt(dot(v, v));
    return {v.x / l, v.y / l, v.z / l};
}

__device__ __host__ vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
    return {a.x * v.x + b.x * v.y + c.x * v.z,
        a.y * v.x + b.y * v.y + c.y * v.z,
        a.z * v.x + b.z * v.y + c.z * v.z};
}

__device__ __host__ void print(vec3 v) {
    printf("%e %e %e\n", v.x, v.y, v.z);
}

__device__ __host__ vec3 operator+(const vec3 &l, const vec3 &r) {
    return {l.x + r.x, l.y + r.y, l.z + r.z};
}

__device__ __host__ vec3 operator-(const vec3 &l, const vec3 &r) {
    return {l.x - r.x, l.y - r.y, l.z - r.z};
}

__device__ __host__ vec3 operator-(const vec3 &l) {
    return {- l.x, - l.y, - l.z};
}

__device__ __host__ vec3 operator*(const float &l, const vec3 &r) {
    return {r.x * l, r.y * l, r.z * l};
}

__device__ __host__ vec3 operator*(const vec3 &l, const float &r) {
    return r * l;
}

__device__ __host__ vec3 operator/(const vec3 &l, const float &r) {
    return {l.x / r, l.y / r, l.z / r};
}

__device__ __host__ vec3 operator*(const vec3 &l, const vec3 &r) {
    return {l.x * r.x, l.y * r.y, l.z * r.z};
}


__device__ __host__ float4 operator+(const float4 &l, const float4 &r) {
    return {(float)min(l.x + r.x, 1.0f), (float)min(l.y + r.y, 1.0f), (float)min(l.z + r.z, 1.0f), 1};
}

__device__ __host__ float4 operator*(const float &l, const float4 &r) {
    return {(float)min(l * r.x, 1.0f), (float)min(l * r.y, 1.0f), (float)min(l * r.z, 1.0f), 1};
}


//Материал, полигон, свет
const int MATERIAL_COUNT = 7;   
const float RADIUS = 0.03f;
const int DETH_REC_SHADE = 7;
int DETH_REC = 10;
int rey_count_cpu = 0;
int err;

struct light {
    vec3 ia;
    vec3 il;
    vec3 pos;
};

struct material {
    vec3 ka;
    vec3 ks;
    vec3 kd;
    float p;  
    float4 color;
    float reflection;
    float refraction;
    float refind; // коэффициент преломления
};

struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    int matidx;
    vec3 bv1;
    vec3 bv2;
    vec3 bp0;
};

struct line {
    vec3 a;
    vec3 b;
};

struct rround {
    vec3 a;
    float r;
};

struct ray {
    vec3 pos;
    vec3 dir;
    int pix_i;
    int pix_j;
    float coef;
};

vector<line> lines;
vector<rround> rounds;
vector<vec3> vertexes;
vector<trig> trigs;
vector<light> lights;
material materials[MATERIAL_COUNT];
uchar4 *text;

__device__ __host__ vec3 textColor(trig trg, uchar4 *text, vec3 point, int textw, int texth) {
    vec3 main = point - trg.bp0;

    //a * v1 + b * v2 = main

    double b = (main.x * trg.bv1.y - main.y * trg.bv1.x) / (trg.bv2.x * trg.bv1.y - trg.bv2.y * trg.bv1.x);
    double a = (main.x * trg.bv2.y - main.y * trg.bv2.x) / (trg.bv1.x * trg.bv2.y - trg.bv1.y * trg.bv2.x);


    return {text[(int)(textw * a) + (int)(texth * b) * textw].x / 255.f, text[(int)(textw * a) + (int)(texth * b) * textw ].y / 255.f, text[(int)(textw * a) + (int)(texth * b)  * textw].z / 255.f};
}

__device__ __host__ float4 phongShade(trig trg, vec3 pos, vec3 dir, vec3 point, vec3 normal, float ts, light lt, vec3 lil, material* materials, uchar4 *text, int textw, int texth) {

    vec3 amb = materials[trg.matidx].ka * lt.ia;

    vec3 lightv = norm(lt.pos - point);
    vec3 dif = materials[trg.matidx].kd * dot(normal, lightv) * lil;

    dif.x = max(dif.x, 0.f);
    dif.y = max(dif.y, 0.f);
    dif.z = max(dif.z, 0.f);

    vec3 spec = materials[trg.matidx].ks;

    vec3 h = 2 * normal * (dot(normal, lightv)) - lightv;
    float s = dot(normal, h);

    for (int i = 0; i < materials[trg.matidx].p; ++i) {
        spec = spec * s;
    }

    spec = spec * lil;

    spec.x = max(spec.x, 0.f);
    spec.y = max(spec.y, 0.f);
    spec.z = max(spec.z, 0.f);

    if (dif.x < 1e-4 && dif.y < 1e-4 && dif.z < 1e-4) {
        spec = {0, 0, 0};
    }

    vec3 v = {0, 0, 0};

    if (trg.matidx == 2) {
        v = textColor(trg, text, point, textw, texth);
        v.x = v.x * materials[trg.matidx].color.x;
        v.y = v.y * materials[trg.matidx].color.y;
        v.z = v.z * materials[trg.matidx].color.z;
    } else {
        v = {(float)materials[trg.matidx].color.x, (float)materials[trg.matidx].color.y, (float)materials[trg.matidx].color.z};
    }
    
    v = (amb + dif + spec) * v;

    return {(float)min(v.x, 1.0f),  (float)min(v.y, 1.0f), (float)min(v.z, 1.0f), 1.0f};
}

//пиксели от 0 до n

__host__ void buildSpace(vec3 center_cube, vec3 color_cube, float radius_cube, float refraction_cube, float reflection_cube, int round_count_cube,
    vec3 center_eco, vec3 color_eco, float radius_eco, float refraction_eco, float reflection_eco, int round_count_eco,
    vec3 center_tetr, vec3 color_tetr, float radius_tetr, float refraction_tetr, float reflection_tetr, int round_count_tetr,
    vector<vec3> pos_floor, vec3 color_floor, float refraction_floor) {

    //Куб грани
    materials[0] = {{0.36, 0.36, 0.36}, {1, 1, 1}, {1, 1, 1}, 23, {0, 0, 0, 1}, 0, 0, 0};
    //Куб стороны
    materials[1] = {{0.36, 0.36, 0.36}, {1, 1, 1}, {1, 1, 1}, 23, {color_cube.x, color_cube.y, color_cube.z, 1}, reflection_cube, refraction_cube, 1};


    //материал для пола, по нему определяем структуру
    materials[2] = {{0.36, 0.36, 0.36}, {1, 1, 1}, {1, 1, 1}, 23, {color_floor.x, color_floor.y, color_floor.z, 1}, 0, 0, 0};


    //Икосаэдр грани
    materials[3] = {{0.36, 0.36, 0.36}, {1, 1, 1}, {1, 1, 1}, 23, {0, 0, 0, 1}, 0, 0, 0};
    //Икосаэдр стороны
    materials[4] = {{0.36, 0.36, 0.36}, {1, 1, 1}, {1, 1, 1}, 3, {color_eco.x, color_eco.y, color_eco.z, 1}, reflection_eco, refraction_eco, 1};


    //Тетраэдр грани
    materials[5] = {{0.36, 0.36, 0.36}, {1, 1, 1}, {1, 1, 1}, 23, {0, 0, 0, 1}, 0, 0, 0};
    //Тетраэдр стороны
    materials[6] = {{0.36, 0.36, 0.36}, {1, 1, 1}, {1, 1, 1}, 3, {color_tetr.x, color_tetr.y, color_tetr.z, 1}, reflection_tetr, refraction_tetr, 1};


    //Пол
    //-5 -5 0  5 -5 0  -5 5 0  5 5 0
    trigs.push_back({pos_floor[0], pos_floor[1], pos_floor[2], 2, {0, 0, 0}, {0, 0, 0}});
    //trigs.push_back({{-5, -5, 0}, {5, -5, 0}, {-5, 5, 0}, 2, {0, 0, 0}, {0, 0, 0}});

    trigs[trigs.size() - 1].bv1 = trigs[trigs.size() - 1].c - trigs[trigs.size() - 1].a;
    trigs[trigs.size() - 1].bv2 = trigs[trigs.size() - 1].b - trigs[trigs.size() - 1].a;
    trigs[trigs.size() - 1].bp0 = trigs[trigs.size() - 1].a;

    //trigs.push_back({{5, -5, 0}, {5, 5, 0}, {-5, 5, 0}, 2, {0, 0, 0}, {0, 0, 0}});
    trigs.push_back({pos_floor[1], pos_floor[3], pos_floor[2], 2, {0, 0, 0}, {0, 0, 0}});

    trigs[trigs.size() - 1].bv1 = trigs[trigs.size() - 1].b - trigs[trigs.size() - 1].a;
    trigs[trigs.size() - 1].bv2 = trigs[trigs.size() - 1].b - trigs[trigs.size() - 1].c;
    trigs[trigs.size() - 1].bp0 = trigs[trigs.size() - 1].c - trigs[trigs.size() - 1].b + trigs[trigs.size() - 1].a;


    FILE * file = fopen("cube.obj", "r");
    if(file == NULL ){
        cout << "File open error\n";
        return;
    } 

    char c = '0';
    int mat = 0;
    while (c != 'e') {

        err = fscanf(file,"%c", &c);

        if (c == 'v') {
            vec3 a = {0, 0, 0};
            err = fscanf(file," %f %f %f\n", &a.x, &a.y, &a.z);

            a.x *= radius_cube;
            a.y *= radius_cube;
            a.z *= radius_cube;

            a.x += center_cube.x;
            a.y += center_cube.y;
            a.z += center_cube.z;

            vertexes.push_back(a);
        }

        if (c == 'u') {
            err = fscanf(file," %d\n", &mat);
        }

        if (c == 'f') {
            int i1, i2, i3;
            err = fscanf(file," %d %d %d\n", &i1, &i2, &i3);
            i1 -= 1;
            i2 -= 1;
            i3 -= 1;
            trigs.push_back({vertexes[i1], vertexes[i2], vertexes[i3], mat, {0, 0, 0}, {0, 0, 0}});
        }

        if (c == 'l') {
            vec3 a = {0, 0, 0};
            vec3 b = {0, 0, 0};
            err = fscanf(file," %f %f %f %f %f %f\n", &a.x, &a.y, &a.z, &b.x, &b.y, &b.z);


            vec3 c = (a + b) / 2.f;
            c = norm(-c) * RADIUS;
            a = a + c;
            b = b + c;


            a.x *= radius_cube;
            a.y *= radius_cube;
            a.z *= radius_cube;

            a.x += center_cube.x;
            a.y += center_cube.y;
            a.z += center_cube.z;


            b.x *= radius_cube;
            b.y *= radius_cube;
            b.z *= radius_cube;

            b.x += center_cube.x;
            b.y += center_cube.y;
            b.z += center_cube.z;

            lines.push_back({a, b});
        }
    }


    for (int i = 0; i < (int)lines.size(); ++i) {

        float dx = abs(lines[i].a.x - lines[i].b.x) / (round_count_cube + 1);
        float dy = abs(lines[i].a.y - lines[i].b.y) / (round_count_cube + 1);
        float dz = abs(lines[i].a.z - lines[i].b.z) / (round_count_cube + 1);

        float x = lines[i].a.x;
        float y = lines[i].a.y;
        float z = lines[i].a.z;

        if (lines[i].a.x > lines[i].b.x) {
            dx *= -1;
        }

        if (lines[i].a.y > lines[i].b.y) {
            dy *= -1;
        }

         if (lines[i].a.z > lines[i].b.z) {
            dz *= -1;
        }


        for (int j = 0; j < round_count_cube; ++j) {
            x += dx;
            y += dy;
            z += dz;

            rounds.push_back({{x, y, z}, RADIUS});
        }
    }

    lines.clear();
    err = fclose(file);


    file = fopen("eco.obj", "r");
    if(file == NULL ){
        cout << "File open error\n";
        return;
    }


    vertexes = vector<vec3>();

    c = '0';
    mat = 0;
    while (c != 'e') {

        err = fscanf(file,"%c", &c);

        if (c == 'v') {
            vec3 a = {0, 0, 0};
            err = fscanf(file," %f %f %f\n", &a.x, &a.y, &a.z);

            a.x *= radius_eco;
            a.y *= radius_eco;
            a.z *= radius_eco;

            a.x += center_eco.x;
            a.y += center_eco.y;
            a.z += center_eco.z;

            vertexes.push_back(a);
        }

        if (c == 'u') {
            err = fscanf(file," %d\n", &mat);
        }

        if (c == 'f') {
            int i1, i2, i3;
            err = fscanf(file," %d %d %d\n", &i1, &i2, &i3);
            i1 -= 1;
            i2 -= 1;
            i3 -= 1;
            trigs.push_back({vertexes[i1], vertexes[i2], vertexes[i3], mat, {0, 0, 0}, {0, 0, 0}});
        }

        if (c == 'l') {
            vec3 a = {0, 0, 0};
            vec3 b = {0, 0, 0};
            err = fscanf(file," %f %f %f %f %f %f\n", &a.x, &a.y, &a.z, &b.x, &b.y, &b.z);


            vec3 c = (a + b) / 2.f;
            c = norm(-c) * RADIUS;
            a = a + c;
            b = b + c;

            a.x *= radius_eco;
            a.y *= radius_eco;
            a.z *= radius_eco;

            a.x += center_eco.x;
            a.y += center_eco.y;
            a.z += center_eco.z;


            b.x *= radius_eco;
            b.y *= radius_eco;
            b.z *= radius_eco;

            b.x += center_eco.x;
            b.y += center_eco.y;
            b.z += center_eco.z;

            lines.push_back({a, b});
        }
    }

    for (int i = 0; i < (int)lines.size(); ++i) {

        float dx = abs(lines[i].a.x - lines[i].b.x) / (round_count_eco + 1);
        float dy = abs(lines[i].a.y - lines[i].b.y) / (round_count_eco + 1);
        float dz = abs(lines[i].a.z - lines[i].b.z) / (round_count_eco + 1);

        float x = lines[i].a.x;
        float y = lines[i].a.y;
        float z = lines[i].a.z;

        if (lines[i].a.x > lines[i].b.x) {
            dx *= -1;
        }

        if (lines[i].a.y > lines[i].b.y) {
            dy *= -1;
        }

         if (lines[i].a.z > lines[i].b.z) {
            dz *= -1;
        }


        for (int j = 0; j < round_count_eco; ++j) {
            x += dx;
            y += dy;
            z += dz;

            rounds.push_back({{x, y, z}, RADIUS});
        }
    }

    lines.clear();


    err = fclose(file);

    file = fopen("tetr.obj", "r");
    if(file == NULL ){
        cout << "File open error\n";
        return;
    }

    vertexes = vector<vec3>();

    c = '0';
    mat = 0;
    while (c != 'e') {

        err = fscanf(file,"%c", &c);

        if (c == 'v') {
            vec3 a = {0, 0, 0};
            err = fscanf(file," %f %f %f\n", &a.x, &a.y, &a.z);

            a.x *= radius_tetr;
            a.y *= radius_tetr;
            a.z *= radius_tetr;

            a.x += center_tetr.x;
            a.y += center_tetr.y;
            a.z += center_tetr.z;

            vertexes.push_back(a);
        }

        if (c == 'u') {
            err = fscanf(file," %d\n", &mat);
        }

        if (c == 'f') {
            int i1, i2, i3;
            err = fscanf(file," %d %d %d\n", &i1, &i2, &i3);
            i1 -= 1;
            i2 -= 1;
            i3 -= 1;
            trigs.push_back({vertexes[i1], vertexes[i2], vertexes[i3], mat, {0, 0, 0}, {0, 0, 0}});
        }

        if (c == 'l') {
            vec3 a = {0, 0, 0};
            vec3 b = {0, 0, 0};
            err = fscanf(file," %f %f %f %f %f %f\n", &a.x, &a.y, &a.z, &b.x, &b.y, &b.z);


            vec3 c = (a + b) / 2.f;
            c = norm(-c) * RADIUS;
            a = a + c;
            b = b + c;

            a.x *= radius_tetr;
            a.y *= radius_tetr;
            a.z *= radius_tetr;

            a.x += center_tetr.x;
            a.y += center_tetr.y;
            a.z += center_tetr.z;


            b.x *= radius_tetr;
            b.y *= radius_tetr;
            b.z *= radius_tetr;

            b.x += center_tetr.x;
            b.y += center_tetr.y;
            b.z += center_tetr.z;

            lines.push_back({a, b});
        }
    }


    
    for (int i = 0; i < (int)lines.size(); ++i) {

        float dx = abs(lines[i].a.x - lines[i].b.x) / (round_count_tetr + 1);
        float dy = abs(lines[i].a.y - lines[i].b.y) / (round_count_tetr + 1);
        float dz = abs(lines[i].a.z - lines[i].b.z) / (round_count_tetr + 1);

        float x = lines[i].a.x;
        float y = lines[i].a.y;
        float z = lines[i].a.z;

        if (lines[i].a.x > lines[i].b.x) {
            dx *= -1;
        }

        if (lines[i].a.y > lines[i].b.y) {
            dy *= -1;
        }

         if (lines[i].a.z > lines[i].b.z) {
            dz *= -1;
        }


        for (int j = 0; j < round_count_tetr; ++j) {
            x += dx;
            y += dy;
            z += dz;

            rounds.push_back({{x, y, z}, RADIUS});
        }
    }

    lines.clear();
    err = fclose(file);
    // for(int i = 0; i < trigs.size(); i++) {
    //     print(trigs[i].a);
    //     print(trigs[i].b);
    //     print(trigs[i].c);
    //     print(trigs[i].a);
    //     printf("\n\n\n");
    // }
    // printf("\n\n\n"); 
}

__device__ __host__ void intersection(vec3 pos, vec3 dir, int &k_min, float &ts_min, trig* trigs, int trigs_count) {

    k_min = -1;
    ts_min = -1;


    for(int k = 0; k < trigs_count; ++k) {

        vec3 e1 = trigs[k].b - trigs[k].a;
        vec3 e2 = trigs[k].c - trigs[k].a;

        vec3 p = prod(dir, e2);
        float div = dot(p, e1);

        if (fabs(div) < 1e-10) {
            continue;
        }

        vec3 t = pos - trigs[k].a;
        float u = dot(p, t) / div;

        if (u < 0.0 || u > 1.0) {
            continue;
        }

        vec3 q = prod(t, e1);

        float v = dot(q, dir) / div;

        if (v < 0.0 || v + u > 1.0) {
            continue;
        }

        float ts = dot(q, e2) / div; 

        if (ts < 0.0) {
            continue;
        }

        if (k_min == -1 || ts < ts_min) {
            k_min = k;
            ts_min = ts;
        }
    }
}

__host__ float4 raytr(vec3 pos, vec3 dir, int d,  uchar4 * text, int textw, int texth) { 
    if (d >= DETH_REC) {
        return {0, 0, 0, 1};
    }
    rey_count_cpu += 1;
    int k_min = -1;
    float ts_min = -1;

    intersection(pos, dir, k_min, ts_min, trigs.data(), trigs.size());

    if (k_min == -1) {
        return {0, 0, 0, 1};
    }

    float4 color = {0, 0, 0, 1};

    vec3 point = pos + ts_min * dir * 0.99999f;


    for (int i = 0; i < (int)rounds.size(); ++i) {

         vec3 v = pos - rounds[i].a;
         float b = 2 * dot(dir, v);
         float c = dot(v, v) - rounds[i].r * rounds[i].r;

         // Находим дискриминант
         float discriminant = (b * b) - (4. * c);

         // Проверяем на мнимые числа
         if (discriminant < 0.0f) {
              continue;
         }

         discriminant = sqrt(discriminant);

         float s0 = (-b + discriminant) / 2.f;
         float s1 = (-b - discriminant) / 2.f;

         // Если есть решение >= 0, луч пересекает сферу
         if ((s0 >= 0.f || s1 >= 0.f) && ts_min > s0 && ts_min > s1) {
                return {1, 1, 1, 1};
         }
    }

    vec3 normal = norm(prod(trigs[k_min].b - trigs[k_min].a, trigs[k_min].c - trigs[k_min].a));

    for (int i = 0; i < (int)lights.size(); ++i) {

        vec3 lil = lights[i].il;

        float ts = -1;
        int kk_min = -1;

        vec3 light_to_point_dir = norm(point - lights[i].pos);
        intersection(lights[i].pos, light_to_point_dir, kk_min, ts, trigs.data(), trigs.size());
        vec3 pointl = lights[i].pos;

        int dd = 0;

        while (kk_min != -1 && dd < DETH_REC_SHADE && kk_min != k_min) { 
            dd += 1;

            vec3 col = {0, 0, 0};

            if (trigs[kk_min].matidx == 2) {
                textColor(trigs[kk_min], text, pointl, textw, texth);
            } else {
                col = {(float)materials[trigs[kk_min].matidx].color.x, (float)materials[trigs[kk_min].matidx].color.y, (float)materials[trigs[kk_min].matidx].color.z };
            }

            lil = lil * materials[trigs[kk_min].matidx].refraction * col;

            if (lil.x < 1e-4 && lil.y < 1e-4 && lil.z < 1e-4) {
                break;
            }


            pointl = pointl + ts * light_to_point_dir  * 1.00001f;
            intersection(pointl, light_to_point_dir, kk_min, ts, trigs.data(), trigs.size());
        }

        color = color + phongShade(trigs[k_min], pos, dir, point, normal, ts_min, lights[i], lil, materials, text, textw, texth);
    }

    if (materials[trigs[k_min].matidx].reflection > 0) {
        vec3 ref = norm(normal * dot(dir, normal) * (-2) + dir);
        color = color + materials[trigs[k_min].matidx].reflection * raytr(point, ref, d + 1, text, textw, texth);
    }

    point = pos + ts_min * dir * 1.00001f; 

    if (materials[trigs[k_min].matidx].refraction > 0) {
        float eta = 1.0f / materials[trigs[k_min].matidx].refind; // eta = in_IOR/out_IOR

        float cos_theta = -dot(normal, dir);

        if (cos_theta < 0) {

            cos_theta *= -1.0f;
            normal = normal * (-1.0f);
            eta = 1.0f / eta;
        }

        float kk = 1.0f - eta * eta * (1.0f - cos_theta * cos_theta);

        if(kk >= 0.0f) {
            vec3 ref = norm( eta * dir + (eta * cos_theta - sqrt(kk)) * normal);
            color = color + materials[trigs[k_min].matidx].refraction * raytr(point, ref, d + 1, text, textw, texth);
        }
    }

    return color;
}

__host__ void render(vec3 pc, vec3 pv, int w, int h, float angle, float4 *data, uchar4 * text, int textw, int texth) {
    int i, j;
    float dw = 2.0f / (w - 1.0f);
    float dh = 2.0f / (h - 1.0f);
    float z = 1.0f / tan(angle * (float)M_PI / 360.0f);
    vec3 bz = norm(pv - pc);
    vec3 bx = norm(prod(bz, {0.0f, 0.0f, 1.0f}));
    vec3 by = norm(prod(bx, bz));


    //Идём по лучам
    for(i = 0; i < w; ++i) { 
        for(j = 0; j < h; ++j) {

            vec3 v = {-1.0f + dw * i, (-1.0f + dh * j) * h / (float)w, z};
            vec3 dir = mult(bx, by, bz, v);
            data[(h - 1 - j) * w + i] = raytr(pc, norm(dir), 0, text, textw, texth);
            // print(pc);
            // print(pc + dir);
            // printf("\n\n\n"); 
        }
    }

    // print(pc);
    // print(pv);
    // printf("\n\n\n"); 
    // for (int i = 0; i < lights.size(); ++i) {
    //     print(lights[i].pos);
    //     print(pv);
    //     printf("\n\n\n");
    // }
}
    
__global__ void kernel_begin_rays(vec3 pc, vec3 pv, int w, int h, float angle, ray *rays, float4* data) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    float dw = 2.0f / (w - 1.0f);
    float dh = 2.0f / (h - 1.0f);
    float z = 1.0f / tan(angle * (float)M_PI / 360.0f);
    vec3 bz = norm(pv - pc);
    vec3 bx = norm(prod(bz, {0.0f, 0.0f, 1.0f}));
    vec3 by = norm(prod(bx, bz));


    for(int i = idx; i < w; i += offsetx) {
        for(int j = idy; j < h; j += offsety) {
            vec3 v = {-1.0f + dw * i, (-1.0f + dh * j) * h / (float)w, z};
            vec3 dir = mult(bx, by, bz, v);
            rays[(h - 1 - j) * w + i] = {pc, norm(dir), i, j, 1};
            data[(h - 1 - j) * w + i] = {0, 0, 0, 1};
        }
    }
}

__global__ void clear_rays2(ray* rays2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        rays2[idx] = {{0, 0, 0}, {0, 0, 0}, -1, -1, -1};
        idx += offset;
    }
}

__global__ void count_rays2(ray* rays2, int n, int *n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {
        if (rays2[idx].pix_i != -1) {
            atomicAdd(n2, 1);
        }

        idx += offset;
    }
}

__global__ void kernel_rays(ray* rays, ray* rays2, float4* data, trig* trigs, rround* rounds, light *lights, material* materials, int n, int w, int h, int trigs_count, int rounds_count, uchar4* text, int textw, int texth, int lights_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    while (idx < n) {

        if (rays[idx].pix_i == -1) {
            idx += offset;
            continue;
        };

        int k_min = -1;
        float ts_min = -1;

        intersection(rays[idx].pos, rays[idx].dir, k_min, ts_min, trigs, trigs_count);

        if (k_min == -1) {
            idx += offset;
            continue;
        }

        vec3 point = rays[idx].pos + ts_min * rays[idx].dir * 0.99999f;

        bool end = false;

        for (int i = 0; i < rounds_count; ++i) {

             vec3 v = rays[idx].pos - rounds[i].a;
             float b = 2 * dot(rays[idx].dir, v);
             float c = dot(v, v) - rounds[i].r * rounds[i].r;

             // Находим дискриминант
             float discriminant = (b * b) - (4. * c);

             // Проверяем на мнимые числа
             if (discriminant < 0.0f) {
                  continue;
             }

             discriminant = sqrt(discriminant);

             float s0 = (-b + discriminant) / 2.f;
             float s1 = (-b - discriminant) / 2.f;


             // Если есть решение >= 0, луч пересекает сферу
            if ((s0 >= 0.f || s1 >= 0.f) && ts_min > s0 && ts_min > s1) {
                    data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i] =  {1, 1, 1, 1};
                    end = true;
                    break;
             }
        }

        if (end) {
            idx += offset;
            continue;
        }

        vec3 normal = norm(prod(trigs[k_min].b - trigs[k_min].a, trigs[k_min].c - trigs[k_min].a));

        for (int i = 0; i < lights_count; ++i) {

            vec3 lil = lights[i].il;

            float ts = -1;
            int kk_min = -1;

            vec3 light_to_point_dir = norm(point - lights[i].pos);
            intersection(lights[i].pos, light_to_point_dir, kk_min, ts, trigs, trigs_count);
            vec3 pointl = lights[i].pos;

            int dd = 0;

            while (kk_min != -1 && dd < DETH_REC_SHADE && kk_min != k_min) { 
                dd += 1;

                vec3 col = {0, 0, 0};
                if (trigs[kk_min].matidx == 2) {
                    textColor(trigs[kk_min], text, pointl, textw, texth);
                } else {
                    col = {(float)materials[trigs[kk_min].matidx].color.x, (float)materials[trigs[kk_min].matidx].color.y, (float)materials[trigs[kk_min].matidx].color.z };
                }


                lil = lil * materials[trigs[kk_min].matidx].refraction * col;

                if (lil.x < 1e-4 && lil.y < 1e-4 && lil.z < 1e-4) {
                    break;
                }
                pointl = pointl + ts * light_to_point_dir  * 1.00001f;
                intersection(pointl, light_to_point_dir, kk_min, ts, trigs, trigs_count);
            }

            float4 ans = rays[idx].coef * phongShade(trigs[k_min], rays[idx].pos, rays[idx].dir, point, normal, ts_min, lights[i], lil, materials, text, textw, texth);

            atomicAdd(&data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].x, ans.x);
            atomicAdd(&data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].y, ans.y);
            atomicAdd(&data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].z, ans.z);

            if (data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].x > 1) {
                data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].x = 1;
            }
            if (data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].y > 1) {
                data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].y = 1;
            }
            if (data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].z > 1) {
                data[(h - 1 - rays[idx].pix_j) * w + rays[idx].pix_i].z = 1;
            }
        }

        if (materials[trigs[k_min].matidx].reflection > 0) {
            vec3 ref = norm(normal * dot(rays[idx].dir, normal) * (-2) + rays[idx].dir);
            rays2[2 * idx] = {point, ref, rays[idx].pix_i, rays[idx].pix_j, rays[idx].coef * materials[trigs[k_min].matidx].reflection};
        }

        point = rays[idx].pos + ts_min * rays[idx].dir * 1.00001f; 

        if (materials[trigs[k_min].matidx].refraction > 0) {

            float eta = 1.0 / materials[trigs[k_min].matidx].refind; // eta = in_IOR/out_IOR

            float cos_theta = -dot(normal, rays[idx].dir);

            if (cos_theta < 0) {

                cos_theta *= -1.0f;
                normal = normal * (-1.0f);
                eta = 1.0f / eta;
            }

            float kk = 1.0f - eta * eta * (1.0f - cos_theta * cos_theta);

            if(kk >= 0.0f) {
                vec3 ref = norm( eta * rays[idx].dir + (eta * cos_theta - sqrt(kk)) * normal);
                rays2[2 * idx + 1] = {point, ref, rays[idx].pix_i, rays[idx].pix_j, rays[idx].coef * materials[trigs[k_min].matidx].refraction};
            }
        }
        idx += offset;
    }
}

__global__ void kernel_compact(ray* rays, ray* rays2, int n, int *idx2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;

    //while()

    while (idx < n) {

        if (rays2[idx].pix_i != -1) {
            rays[atomicAdd(idx2, 1)] = rays2[idx];       
        }

        idx += offset;
    }
}


__global__ void kernel_ssaa(float4* data, float4* sdata, int w, int h, int ssaacoef) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for(int i = idx; i < w; i += offsetx) {
        for(int j = idy; j < h; j += offsety) {

            vec3 col = {0, 0, 0};

            for (int i2 = i * ssaacoef; i2 < i * ssaacoef + ssaacoef; ++i2) {
                for (int j2 = j * ssaacoef; j2 < j * ssaacoef + ssaacoef; ++j2) {

                    col.x = col.x + sdata[(h * ssaacoef - 1 - j2) * w * ssaacoef + i2].x;
                    col.y = col.y + sdata[(h * ssaacoef - 1 - j2) * w * ssaacoef + i2].y;
                    col.z = col.z + sdata[(h * ssaacoef - 1 - j2) * w * ssaacoef + i2].z;

                }
            }

             data[(h - 1 - j) * w + i] = {(float)(col.x / (float)(ssaacoef * ssaacoef)), ((float)col.y / (float)(ssaacoef * ssaacoef)), ((float)col.z / (float)(ssaacoef * ssaacoef)), 1};
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc != 1 && argv[1][2] == 'd') {  
        printf("150\nres/%%d.data\n1920 1080\n6 5 1.5707963267948966 0 2 0 2.0053522829578814 1.0026761414789407 0 0\n3 0 4.71238898038469 0 0 0 0 1.0026761414789407 0 0\n-2 0 2 0.1 0.6 0 1 0.7 1 5\n0 3 3 0.5 0.5 0.5 1 0.5 1 2\n3 -3 1 0.7 0 0 1.5 0.7 1 4\n-5 -5 0 5 -5 0 -5 5 0 5 5 0\ntext.data\n1 1 1 0\n1\n0 0 14 1 1 1\n5 2");
        return 0;
    }

    int cadrs;
    cin >> cadrs;

    string way;
    cin >> way;

    int w, h;

    cin >> w >> h;


    float r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc, r0n, z0n, phi0n, Arn, Azn;
    float wrn, wzn, wphin, prn, pzn;

    cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc >> r0n >> z0n >> phi0n >> Arn >> Azn;
    cin >> wrn >> wzn >> wphin >> prn >> pzn;



    vec3 center_cube;
    cin >> center_cube.x >> center_cube.y >> center_cube.z;

    vec3 color_cube;
    cin >> color_cube.x >> color_cube.y >> color_cube.z;

    float radius_cube;
    cin >> radius_cube;

    float refraction_cube, reflection_cube;
    cin >> refraction_cube >> reflection_cube;

    int round_count_cube;
    cin >> round_count_cube;



    vec3 center_eco;
    cin >> center_eco.x >> center_eco.y >> center_eco.z;

    vec3 color_eco;
    cin >> color_eco.x >> color_eco.y >> color_eco.z;


    float radius_eco;
    cin >> radius_eco;

    float refraction_eco, reflection_eco;
    cin >> refraction_eco >> reflection_eco;

    int round_count_eco;
    cin >> round_count_eco;



    vec3 center_tetr;
    cin >> center_tetr.x >> center_tetr.y >> center_tetr.z;

    vec3 color_tetr;
    cin >> color_tetr.x >> color_tetr.y >> color_tetr.z;

    float radius_tetr;
    cin >> radius_tetr;

    float refraction_tetr, reflection_tetr;
    cin >> refraction_tetr >> reflection_tetr;

    int round_count_tetr;
    cin >> round_count_tetr;


    vector<vec3> pos_floor(4);
    for (int i = 0; i < 4; ++i) {
        vec3 pos_f;
        cin >> pos_f.x >> pos_f.y >> pos_f.z;
        pos_floor[i] = pos_f;
    }

    string text_name;
    cin >> text_name;

    FILE *textf;

    if ((textf = fopen(text_name.c_str(), "rb")) == NULL) {
        cout << "File open error\n";
        return -1;
    }

    int textw = 0;
    int texth = 0;

    err = fread(&textw, sizeof(int), 1, textf);
    err = fread(&texth, sizeof(int), 1, textf);

    uchar4 * text = (uchar4 *)malloc(sizeof(uchar4) * textw * texth);
    err = fread(text, sizeof(uchar4), textw * texth, textf);

    err = fclose(textf);


    vec3 color_floor;
    cin >> color_floor.x >> color_floor.y >> color_floor.z;

    float refraction_floor;
    cin >> refraction_floor;


    int lights_count;
    cin >> lights_count;

    for (int i = 0; i < lights_count; ++i) {
        vec3 pos_light;
        cin >> pos_light.x >> pos_light.y >> pos_light.z;
        vec3 color_light;
        cin >> color_light.x >> color_light.y >> color_light.z;
        lights.push_back({(vec3){1, 1, 1}, color_light, pos_light});
    }

    cin >> DETH_REC;


    int ssaacoef = 1;
    cin >> ssaacoef;

    char buff[256];
    float4 *data = (float4*)malloc(sizeof(float4) * w * h);
    float4 *sdata = (float4*)malloc(sizeof(float4) * w * h * ssaacoef * ssaacoef);
    vec3 pc, pv;

    buildSpace(center_cube, color_cube, radius_cube, refraction_cube, reflection_cube,round_count_cube,
    center_eco, color_eco, radius_eco, refraction_eco, reflection_eco,round_count_eco,
    center_tetr, color_tetr, radius_tetr, refraction_tetr, reflection_tetr,round_count_tetr,
    pos_floor, color_floor, refraction_floor);

    float4 *dev_sdata;
    float4 *dev_data;

    ray *dev_rays;
    ray *dev_rays2;
    ray* rays;
    ray* rays2;
    trig *dev_trigs;
    rround *dev_rounds;
    light *dev_lights;
    material *dev_materials;
    int *dev_n2;
    int *dev_idx2;
    uchar4 *dev_text;

    int n = w * h * ssaacoef * ssaacoef;

    rays2 = (ray*)malloc(sizeof(ray) * n * 2);
    rays = (ray*)malloc(sizeof(ray) * n);

    if (argc == 1 || argv[1][2] == 'g') {

        
        CSC(cudaMalloc(&dev_rays, sizeof(ray) * n));
        CSC(cudaMemcpy(dev_rays, rays, sizeof(ray) * n, cudaMemcpyHostToDevice));

        
        CSC(cudaMalloc(&dev_rays2, sizeof(ray) * n * 2));
        CSC(cudaMemcpy(dev_rays2, rays2, sizeof(ray) * n * 2, cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&dev_sdata, sizeof(float4) * n));
        CSC(cudaMemcpy(dev_sdata, sdata, sizeof(float4) * n, cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&dev_data, sizeof(float4) * w * h));
        CSC(cudaMemcpy(dev_data, data, sizeof(float4) * w * h, cudaMemcpyHostToDevice));
        
        CSC(cudaMalloc(&dev_trigs, sizeof(trig) * trigs.size()));
        CSC(cudaMemcpy(dev_trigs, trigs.data(), sizeof(trig) * trigs.size(), cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&dev_rounds, sizeof(rround) * rounds.size()));
        CSC(cudaMemcpy(dev_rounds, rounds.data(), sizeof(rround) * rounds.size(), cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&dev_lights, sizeof(light) * lights.size()));
        CSC(cudaMemcpy(dev_lights, lights.data(), sizeof(light) * lights.size(), cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&dev_materials, sizeof(material) * MATERIAL_COUNT));
        CSC(cudaMemcpy(dev_materials, materials, sizeof(material) * MATERIAL_COUNT, cudaMemcpyHostToDevice));

        CSC(cudaMalloc(&dev_text, sizeof(uchar4) * textw * texth));
        CSC(cudaMemcpy(dev_text, text, sizeof(uchar4) * textw * texth, cudaMemcpyHostToDevice));


    }


    float dt = 2 * (float)M_PI / (float)cadrs;

    for(int k = 0; k < cadrs; k++) { 

        printf("%d\t", k);

        float r = r0c + Arc * sin(wrc * dt * k + prc);
        float phi = phi0c + wphic * dt * k;
        pc = (vec3) {r * cos(phi), r * sin(phi), z0c + Azc * sin(wzc * k * dt + pzc)};

        r = r0n + Arn * sin(wrn * dt * k + prn);
        phi = phi0n + wphin * k * dt;
        pv = (vec3) {r * cos(phi), r * sin(phi), z0n + Azn * sin(wzn * k * dt + pzn)};

        

        if (argc == 1 || argv[1][2] == 'g') {  

            cudaEvent_t start, stop;
            CSC(cudaEventCreate(&start));
            CSC(cudaEventCreate(&stop));
            CSC(cudaEventRecord(start));

            kernel_begin_rays<<<dim3(1, 16), dim3(1, 32)>>>(pc, pv, w * ssaacoef, h * ssaacoef, 120.0f, dev_rays, dev_sdata);
            CSC(cudaGetLastError());

            int n2 = 0;
            CSC(cudaMalloc(&dev_n2, sizeof(int)));
            CSC(cudaMalloc(&dev_idx2, sizeof(int)));

            for (int i = 0; i < DETH_REC; ++i) {

                if (n2 > n) {
                    n = n * 2;
                    CSC(cudaFree(dev_rays2));
                    CSC(cudaMalloc(&dev_rays2, sizeof(ray) * n * 2));
                }

                clear_rays2<<<512, 512>>>(dev_rays2, n * 2);

                CSC(cudaGetLastError());


                kernel_rays<<<512, 512>>>(dev_rays, dev_rays2, dev_sdata, dev_trigs, dev_rounds, dev_lights, dev_materials, n, w * ssaacoef, h * ssaacoef, trigs.size(), rounds.size(), dev_text, textw, texth, lights.size());
                CSC(cudaGetLastError());

                n2 = 0;
                CSC(cudaMemcpy(dev_n2, &n2, sizeof(int), cudaMemcpyHostToDevice));
                count_rays2<<<512, 512>>>(dev_rays2, n * 2, dev_n2);
                CSC(cudaGetLastError());


                CSC(cudaMemcpy(&n2, dev_n2, sizeof(int), cudaMemcpyDeviceToHost));
                
                if (n2 > n) {
                    CSC(cudaFree(dev_rays));
                    CSC(cudaMalloc(&dev_rays, sizeof(ray) * n * 2));
                    clear_rays2<<<512, 512>>>(dev_rays, n * 2);
                } else {
                    clear_rays2<<<512, 512>>>(dev_rays, n);
                }

                CSC(cudaMemset(dev_idx2, 0, sizeof(int)));

                kernel_compact<<<512, 512>>>(dev_rays, dev_rays2, n * 2, dev_idx2);
                CSC(cudaGetLastError());
            }

            CSC(cudaEventRecord(stop));
            CSC(cudaEventSynchronize(stop));
            float t;
            CSC(cudaEventElapsedTime(&t, start, stop));
            CSC(cudaEventDestroy(start));
            CSC(cudaEventDestroy(stop));

            printf("%f\t", t);
            printf("%d\n", n2);

            kernel_ssaa<<<dim3(1, 16), dim3(1, 32)>>>(dev_data, dev_sdata, w, h, ssaacoef);
            CSC(cudaMemcpy(data, dev_data, sizeof(float4) * w * h, cudaMemcpyDeviceToHost));
        }

        if (argc != 1 && argv[1][2] == 'c') {
            rey_count_cpu = 0;
            unsigned int start_time =  clock();

            render(pc, pv, w * ssaacoef, h * ssaacoef, 120.0f, sdata, text, textw, texth);
             
            //SSAA
            for(int i = 0; i < w; ++i) { 
                for(int j = 0; j < h; ++j) {

                    vec3 col = {0, 0, 0};

                    for (int i2 = i * ssaacoef; i2 < i * ssaacoef + ssaacoef; ++i2) {
                        for (int j2 = j * ssaacoef; j2 < j * ssaacoef + ssaacoef; ++j2) {

                            col.x = col.x + sdata[(h * ssaacoef - 1 - j2) * w * ssaacoef + i2].x;
                            col.y = col.y + sdata[(h * ssaacoef - 1 - j2) * w * ssaacoef + i2].y;
                            col.z = col.z + sdata[(h * ssaacoef - 1 - j2) * w * ssaacoef + i2].z;

                        }
                    }

                    data[(h - 1 - j) * w + i] = {(float)(col.x / (float)(ssaacoef * ssaacoef)), ((float)col.y / (float)(ssaacoef * ssaacoef)), ((float)col.z / (float)(ssaacoef * ssaacoef)), 1};
                }
            }

            unsigned int end_time = clock(); 
            unsigned int search_time = end_time - start_time; 
            printf("%g ms\t", (double)search_time / 1000);
            printf("%d\n", rey_count_cpu);
        }

        
        uchar4 *datachar = (uchar4*)malloc(sizeof(uchar4) * w * h);

        for (int i = 0; i < w; ++i) {
            for (int j = 0; j < h; ++j) {
                datachar[(h - 1 - j) * w + i].x = (unsigned char)(data[(h - 1 - j) * w + i].x * 255.f);
                datachar[(h - 1 - j) * w + i].y = (unsigned char)(data[(h - 1 - j) * w + i].y * 255.f);
                datachar[(h - 1 - j) * w + i].z = (unsigned char)(data[(h - 1 - j) * w + i].z * 255.f);
                datachar[(h - 1 - j) * w + i].w = (unsigned char)255;
            }
        }


        sprintf(buff, way.c_str(), k); 

        FILE *out; 

        if ((out = fopen(buff, "wb")) == NULL) {
            cout << "File open error\n";
            return -1;
        }

        err = fwrite(&w, sizeof(int), 1, out);
        err = fwrite(&h, sizeof(int), 1, out);    
        err = fwrite(datachar, sizeof(uchar4), w * h , out);
        fclose(out);
    }

    if (argc == 1 || argv[1][2] == 'g') { 

        CSC(cudaFree(dev_data));
        CSC(cudaFree(dev_sdata));
        CSC(cudaFree(dev_n2));
        CSC(cudaFree(dev_idx2));
        CSC(cudaFree(dev_rays));
        CSC(cudaFree(dev_rays2));
        CSC(cudaFree(dev_trigs));
        CSC(cudaFree(dev_lights));
        CSC(cudaFree(dev_materials));
        CSC(cudaFree(dev_rounds));
       
    }

    free(rays);
    free(rays2);
    free(text);
    free(data);
    free(sdata);
    return 0;
}
