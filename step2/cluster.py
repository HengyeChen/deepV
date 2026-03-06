import pandas as pd

def is_point_in_v_region(point, v_center):
    x0, y0 = v_center['x_value'], v_center['y_value']
    x, y = point['x_value'], point['y_value']

    x_left = x - (100 - y) / 2
    x_right = x + (100 - y) / 2

    A = (x, y)
    B = (x_left, 100)
    C = (x_right, 100)

#    print(A, B, C)

    def cross_product(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    cp1 = cross_product(A, B, (x0, y0))
    cp2 = cross_product(B, C, (x0, y0))
    cp3 = cross_product(C, A, (x0, y0))

    return (cp1 >= 0 and cp2 >= 0 and cp3 >= 0) or (cp1 <= 0 and cp2 <= 0 and cp3 <= 0)

def assign_v_clusters(point_file):
    df = pd.read_csv(point_file, sep='\t')
    grouped = df.groupby('image')
    result_dfs = []
    
    for image, group in grouped:
        # 排序后重置索引（关键修复：使用0,1,2...的连续索引）
        sorted_group = group.sort_values('y_value').reset_index(drop=True)
        n = len(sorted_group)
        # 按连续索引初始化（0到n-1）
        cluster_assignments = [[] for _ in range(n)]
        current_cluster = 0
        
        # 用位置索引i遍历（0到n-1）
        for i in range(n):
            point_row = sorted_group.iloc[i]
            point = {
                'x_value': point_row['x_value'],
                'y_value': point_row['y_value']
            }
            
            # 检查当前点是否已分配聚类
            if not cluster_assignments[i]:
                cluster_assignments[i].append(str(current_cluster))
                
                # 遍历其他点（用位置索引j）
                for j in range(n):
                    if i != j:
                        v_center_row = sorted_group.iloc[j]
                        v_center = {
                            'x_value': v_center_row['x_value'],
                            'y_value': v_center_row['y_value']
                        }
                        
                        if is_point_in_v_region(point, v_center):
                            cluster_assignments[j].append(str(current_cluster))
              
#                        print('point', point)
#                        print('v_center', v_center)
#                        print(is_point_in_v_region(point, v_center))
#                        print('current_cluster', current_cluster)

                current_cluster += 1
        
        sorted_group['cluster_v_region'] = [
            ','.join(clusters) if clusters else '' 
            for clusters in cluster_assignments
        ]
        result_dfs.append(sorted_group)
    
    result_df = pd.concat(result_dfs, ignore_index=True)
    return result_df

if __name__ == "__main__":
    point_file = "cluster.image9.csv" 
    result_df = assign_v_clusters(point_file)
    print(result_df[['image', 'point', 'cluster_v_region']])
