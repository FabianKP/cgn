# This aspects have to be incorporated in future versions of cgn to make it less embarassing.

- There should be a way to remove constraints from cgn.Problem instances.

- There should be an easier way to modify the regularization. If I want to change, say, a regularization parameter, 
  I currently have to call .set_regularization and provide the old mean and regularization operator. 
  This should be easier.

- Remove dependencies by implementing subroutines yourself (in C++).

- Write a problem template, so that the user knows how to use cgn and basically only has to fill in his stuff.
