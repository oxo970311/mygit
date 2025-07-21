import random

from keras.src.backend.jax.nn import binary_crossentropy

# list = []
#
# for i in range(10):
#     list.append(random.randrange(1, 100))
#
# print(list)
#
# def swap(a, b):
#     temp = a
#     a = b
#     b = temp
#
#
# def bubble_sort(list):
#     n = len(list)
#     for i in range(n - 1):  # Outer loop for passes
#         for j in range(n - 1 - i):
#             if list[j + 1] < list[j]:
#                 swap(list[j + 1], list[j])
#                 # list[j], list[j + 1] = list[j + 1], list[j]
#
#     return list
#
#
# print(bubble_sort(list))
#
# class ListNode:
#     def __init__(self,x):
#         self.val = x
#         self.next= None
#
#
# class Solution:
#
#     def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
#         if l1 and l2:
#             if l1.val > l2.val:
#                 l1, l2 = l2, l1
#             l1.next = self.margeTwoLists(l1.next, l2)
#
#         return l1 or l2
#
#     def sortList(self, head: ListNode) -> ListNode:
#         if not (head and head.next):
#             return head
#
#         half, slow, fast = None, head, head
#         while fast and fast.next:
#             half, slow, fast = slow, slow.next, fast.next.next
#         half.next = None
#
#         l1 = self.sortList(head)
#         l2 = self.sortList(slow)
#
#         return self.mergeTwoList(l1, l2)
#
#
# s = Solution()
# print(s.mergeTwoLists(None, ListNode))

# nums = []
#
# for i in range(5):
#     nums.append(random.randrange(1, 50))
#
# print(nums)
#
# split_digit = [int(digit) for num in nums for digit in str(num)]
# print(split_digit)
#
# size = len(split_digit)
# print(size)
#
#
# c = []
# def sort(nums):
#
#
#     for k in range(size - 1):
#          for i in range(0, size - 1):
#             if nums[i] < nums[i + 1]:
#                 nums[i], nums[i + 1] = nums[i + 1], nums[i]
#
#     return nums
#
# print(sort(split_digit))
# print(c)

# # 난수
# import random
#
# random.seed(42)
#
# num = int(input("put in number under 100 : "))
#
# # 리스트 선언
# arr = []
# even = []
# odd = []
#
# # 랜덤값 1~99 20개
# arr = random.sample(range(1, 100), num)
#
# n = len(arr)
# # 버블 정렬
# def sort(arr):
#     for i in range(0, n - 1):
#         for j in range(0, n - 1 - i):
#             if arr[j + 1] < arr[j]:
#                 arr[j + 1], arr[j] = arr[j], arr[j + 1]
#     return arr
#
# # 짝수, 홀수 구분
# def odd_even(arr):
#     for j in range(0, len(arr)):
#
#         if arr[j] % 2 == 0:
#             even.append(arr[j])
#
#
#         elif arr[j] % 2 == 1:
#             odd.append(arr[j])
#
# #출력
# print(arr)
# print(sort(arr))
# print(odd_even(arr))
# print("even list : ", even)
# print("odd list : ", odd)

# nums = [-1, 0, 3, 5, 9, 12]
#
#
# def search1(nums, target):
#     def binary_search(left, right):
#         if left <= right:
#             mid = (left + right) // 2
#
#             if nums[mid] < target:
#                 return binary_search(mid + 1, right)
#             elif nums[mid] > target:
#                 return binary_search(left, mid - 1)
#             else:
#                 return mid
#         else:
#             return -1
#
#     return binary_search(0, len(nums) - 1)
#
#
# print(search1(nums, 3))
#
#
# def search2(nums, target):
#     left, right = 0, len(nums) - 1
#     while left <= right:
#         mid = (left + right) // 2
#
#         if nums[mid] < target:
#             left = mid + 1
#         elif nums[mid] > target:
#             right = mid - 1
#         else:
#             return mid
#     else:
#         return -1
#
# print(search2(nums, 8))


# 자리 정렬
import random
# random.seed(42)
# random_ = random.sample(range(1, 50 + 1), 25)
# list_data = [random_[i:i + 5] for i in range(0, 25, 5)]
#
# for arr in list_data:
#     print(arr)
#
#
# def search():
#     while (1):
#         num = int(input("choice number : "))
#         for i in range(5):
#             for j in range(5):
#                 if list_data[i][j] == num:
#                     print("found : ", num)
#                     return 0
#
# search()
import random
binary_list = random.sample(range(1, 50 + 1), 25)
binary_list.sort()

def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # 값의 인덱스 반환
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # 값이 없을 때

# 찾는고자 하는 숫자 입력
def binary():
    while (1):
        num = int(input("choice number 1 ~ 50 : "))
        index = binary_search(binary_list, num)

        if index != -1:
            print(f" {num} found at index {index}")
            return 0
        else:
            print(f" {num} not found in the list.")
            continue

binary()
print("list : ", binary_list)