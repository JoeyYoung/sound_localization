#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <tuple>
#include <string>

#define PI 3.1415927

using namespace std;

typedef struct tree_node {
    /* the range of this node in the map */
    float t_range;
    float b_range;
    float r_range;
    float l_range;

    /* the radius of cuurrnt node */
    float radius;

    /* parent node */
    struct tree_node* parent;

    /* 
        four children inited with null
        right-top; left-top ...
     */
    struct tree_node* rt;
    struct tree_node* lt;
    struct tree_node* rb;
    struct tree_node* lb;

    bool occupied;

} *TreeNode;

typedef struct lidar_node {
    /* three elements read from lidar */
    float theta;
    float dist;
    float quality;

    struct lidar_node* next;

} *LidarNode;


class QuadTree{
    public:
        TreeNode root;

        /* current position of walker */
        float walker_x;
        float walker_y;
        float walker_theta;

        /* build initial tree with only root node, and specific settings */
        QuadTree(TreeNode t):root(t){            
            /* area is 20 x 20 meters */
            root->radius = 40.0;
            
            root->t_range = 40.0;
            root->b_range = 0.0;
            root->r_range = 40.0;
            root->l_range = 0.0;    

            printf("Quad Tree is inited with one root node ... \n");

            /* init walker info */
            walker_x = 0.0;
            walker_y = 0.0;
            walker_theta = 0.0;
        }

        ~QuadTree(){
            free(root);
        };

        /* if necessary, update corresponding node based on lidar info list */
        void update_tree(LidarNode header, float cur_walker_x, float cur_walker_y, float cur_theta){
            printf("=== start update tree with lidar info & odmetry info ...\n");
            
            update_walker_pos(cur_walker_x, cur_walker_y, cur_theta);

            LidarNode p = header;
            while(p -> next != NULL){
                p = p -> next;

                /* transfer from dist-theta to x-y */
                float* x_y = transfer_to_point(p->theta, p->dist);
                float x = x_y[0];
                float y = x_y[1];

                /* nothing detected */
                if(x == walker_x && y == walker_y)
                    continue;

                TreeNode node = find_which_node(root, x, y);
                printf("\t=== find point (%f, %f) in leaf node with radius: %f\n", x, y, node->radius);
                extend_children(node, x, y);
            }

            printf("=== update tree done. \n");
        }

        // TODO:
        /* show map scanned, show tree constructed */
        void visualize_tree(TreeNode node){
            /* print out layer by layer, need queue */
        }

        /* save x, y points to file */
        void save_points(LidarNode header, char* filename){
            FILE *fp = NULL;
            fp = fopen(filename, "w+");
            LidarNode p = header;

            while(p -> next != NULL){
                p = p -> next;

                /* transfer from dist-theta to x-y */
                float* x_y = transfer_to_point(p->theta, p->dist);
                float x = x_y[0];
                float y = x_y[1];

                char out[100];
                sprintf(out, "%f;%f\n", x, y);
                /* store into file */

                fprintf(fp, out);
            }

            fclose(fp);
        }

    private:
        /* begin from arbitrary node to extend */
        void extend_children(TreeNode node, float x, float y){
            /* can not extend */
            if(node -> radius < 0.5){
                printf("\t\textended to minimum unit; \n");
                return;
            }

            printf("\t\textend children for node with radius: %f \n", node->radius);
            
            /* split four children */
            TreeNode rt = (TreeNode)malloc(sizeof(struct tree_node));
            memset(rt, 0, sizeof(struct tree_node));
            rt->parent = node;
            rt->radius = node->radius / 2;
            rt->t_range = node->t_range;
            rt->b_range = (node->t_range + node->b_range) / 2;
            rt->l_range = (node->l_range + node->r_range) / 2;
            rt->r_range = node->r_range;

            TreeNode lt = (TreeNode)malloc(sizeof(struct tree_node));
            memset(lt, 0, sizeof(struct tree_node));
            lt->parent = node;
            lt->radius = node->radius / 2;
            lt->t_range = node->t_range;
            lt->b_range = (node->t_range + node->b_range) / 2;
            lt->l_range = node->l_range;
            lt->r_range = (node->l_range + node->r_range) / 2;           
            
            TreeNode lb = (TreeNode)malloc(sizeof(struct tree_node));
            memset(lb, 0, sizeof(struct tree_node));
            lb->parent = node;
            lb->radius = node->radius / 2;
            lb->t_range = (node->t_range + node->b_range) / 2;
            lb->b_range = node->b_range;
            lb->l_range = node->l_range;
            lb->r_range = (node->l_range + node->r_range) / 2;

            TreeNode rb = (TreeNode)malloc(sizeof(struct tree_node));
            memset(rb, 0, sizeof(struct tree_node));
            rb->parent = node;
            rb->radius = node->radius / 2;
            rb->t_range = (node->t_range + node->b_range) / 2;
            rb->b_range = node->b_range;
            rb->l_range = (node->l_range + node->r_range) / 2;
            rb->r_range = node->r_range;

            /* link children */
            node->lt = lt;
            node->rt = rt;
            node->rb = rb;
            node->lb = lb;

            /* according to which children, call extend further */
            TreeNode belong_node = find_which_node(node, x, y);
            extend_children(belong_node, x, y);
        }
        
        /* begin from one node, find leaf node belongs to */
        TreeNode find_which_node(TreeNode node, float point_x, float point_y){
            node->occupied = true;

            /* find until leaf node */
            if (node->rt == NULL){
                return node;
            }

            // printf("node l_range: %f\n", node->l_range);
            // printf("node r_range: %f\n", node->r_range);
            // printf("node t_range: %f\n", node->t_range);
            // printf("node b_range: %f\n", node->b_range);

            if(node->l_range < point_x && point_x <= (node->l_range + node->r_range) / 2 && (node->b_range + node->t_range) / 2 < point_y && point_y < node->t_range){
                // printf("left top part \n");
                return find_which_node(node->lt, point_x, point_y);
            }else if((node->l_range + node->r_range) / 2 < point_x && point_x <= node->r_range && (node->b_range + node->t_range) / 2 < point_y && point_y < node->t_range){
                // printf("right top part \n");
                return find_which_node(node->rt, point_x, point_y);
            }else if(node->l_range < point_x && point_x <= (node->l_range + node->r_range) / 2 && node->b_range < point_y && point_y <= (node->b_range + node->t_range) / 2){
                // printf("left below part \n");
                return find_which_node(node->lb, point_x, point_y);
            }else{
                // printf("right below part \n");
                return find_which_node(node->rb, point_x, point_y);
            }
                
        }

        float* transfer_to_point(float theta, float dist){
            if(theta + walker_theta < 360){
                theta = theta + walker_theta;
            }else{
                theta = theta + walker_theta - 360.0;
            }
            static float x_y[2];
            if(theta == 0 || theta == 360){
                x_y[0] = walker_x;
                x_y[1] = walker_y + dist/1000.0;
            }else if(theta > 0 && theta < 90){
                x_y[0] = walker_x + dist/1000.0 * sin(theta * PI / 180);
                x_y[1] = walker_y + dist/1000.0 * cos(theta * PI / 180);
            }else if(theta == 90){
                x_y[0] = walker_x + dist/1000.0;
                x_y[1] = walker_y;
            }else if(theta > 90 && theta < 180){
                float theta_ = theta - 90;
                x_y[0] = walker_x + dist/1000.0 * cos(theta_ * PI / 180);
                x_y[1] = walker_y - dist/1000.0 * sin(theta_ * PI / 180);
            }else if(theta == 180){
                x_y[0] = walker_x;
                x_y[1] = walker_y - dist/1000.0;
            }else if(theta > 180 && theta < 270){
                float theta_ = theta - 180;
                x_y[0] = walker_x - dist/1000.0 * sin(theta_ * PI / 180);
                x_y[1] = walker_y - dist/1000.0 * cos(theta_ * PI / 180);
            }else if(theta == 270){
                x_y[0] = walker_x - dist/1000.0;
                x_y[1] = walker_y;
            }else{
                // 270 - 360
                float theta_ = theta - 270;
                x_y[0] = walker_x - dist/1000.0 * cos(theta_ * PI / 180);
                x_y[1] = walker_y + dist/1000.0 * sin(theta_ * PI / 180);
            }
            return x_y;
        }

        /* update walker position based on odometry */
        void update_walker_pos(float x, float y, float theta){
            walker_x = x;
            walker_y = y;
            walker_theta = theta;

            printf("walker position updated ... \n");
        }
};